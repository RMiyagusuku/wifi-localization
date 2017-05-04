#include <Python.h>
#include <math.h>

#include <levmar.h>
#include <float.h>
#include <sys/time.h>                // for gettimeofday()

#define max_size 1000
#define max_ap 1000
#define eps 1e-3
#define debug 1
#define sigmoidk 50

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

/* Python wraper for levmar function to fit model z = p0 - p1log((x-p2)^2+(y-p3)^2) */
FILE *f;
FILE *f2;

/* structure for passing data */
typedef struct xtradata{
    double x[max_size];
    double y[max_size];
    double x2y2[max_size]; //x^2+y^2 computed to optimize distance to ap evaluation
    double z[max_size][max_ap];
    int Nnonzero[max_ap];
    int Nzero[max_ap];
    int n, nap, ap, m;
} xtradata;

/* Function prototypes*/
void distance_squared(double *d2, double *p, void *data);
void modelfun(double *p, double *e, int m, int n, void *data);
void jacmodelfunp01(double *p, double *jac, int m, int n, void *data);   
void jacmodelfunp23(double *p, double *jac, int m, int n, void *data);   
void irls(double *p, int m, int n, void *data);
void init_ap(double *p, void *data);
void update_p(double *p, void *data);
void solve_linear(double *p, void *data);
int load_variables(xtradata *data, PyObject *args);
double sigmoid(double x);
int sign(double x);
double clip(double x, double xmin, double xmax);

/* Functions*/

double clip(double x, double xmin, double xmax){
    if(x > xmax){return xmax;}
    else{
        if(x<xmin){return xmin;}
        else{return x;}
    }
}

double sigmoid(double x){
    return 1/(1+exp(-sigmoidk*x));
}

int sign(double x){
    if(x==0){return 0;}
    else{
        if(x>0){return 1;}
        else{return -1;}
    }

}

void distance_squared(double *d2, double *p, void *data){
/* efficient squared distance calculation d2 = (x-p2)^2+(y-p3)^2 */
    double twop2, twop3, p22p32;
    int n;
    register int i;
    xtradata *dat;
    dat = (xtradata *)data;
    n = dat->n;
    /* d2 = x^2-2*x2*p2 + p2^2 + y^2-2*y2*p3 + p3^2 + eps
       eps is to avoid d2 equal zero
       For speed:
       x^2+y^2 pre computed in dat->x2y2
       2*p2, 2*p3 and p2^2+p3^2 computed once and stored in twop2, twop3 and p22p32
   */
    p22p32 = pow(p[2],2)+pow(p[3],2) + eps; 
    twop2 = 2*p[2];
    twop3 = 2*p[3];

    for(i=0; i<n; ++i){
        d2[i] = dat->x2y2[i] + p22p32 - dat->x[i]*twop2 - dat->y[i]*twop3;
    } 
}

void modelfun(double *p, double *e, int m, int n, void *data){
    /* error model function E = [sign(z)+(1-sign(z))*sigmoid(zp)]*(zp-z)^2 
                           zp = p0-p1log(d2)        d2 = (x-p2)^2+(y-p3)^2 
    */
    register int i;
    double *d2 = (double *) PyMem_Malloc(sizeof(double)*n);
    double zp, sigmoid_zp; //w=0;    
    int ap;

    xtradata *dat;
    dat = (xtradata *)data;
    ap  = dat->ap;
    //TODO: fix pzero
    //pzero    = 1.0; //Pzero*dat->Nnonzero[ap]/(dat->Nzero[ap]+0.01);

    distance_squared(d2,p,data); 
    for(i=0; i<n; ++i){
        zp   = p[0]-p[1]*log(d2[i]);
        if(sign(dat->z[i][ap])==1){
            e[i] = zp-dat->z[i][ap];
        }else{
            sigmoid_zp = sigmoid(zp);
            e[i] = sigmoid_zp*(zp-dat->z[i][ap]);
        }
    }
    if(debug){fprintf(f, "%.3f %.3f %.3f %.3f %.3f \n", p[0], p[1], p[2], p[3], 1.);}
    PyMem_Free(d2);
}

void jacmodelfunp23(double *p, double *jac, int m, int n, void *data){   
/* Jacobian of modelfun() with first and second parameters fixed */
    register int i, j;
    double *d2 = (double *) PyMem_Malloc(sizeof(double)*n);
    double cte, zp, sigmoid_zp; //w=0;
    int ap;

    xtradata *dat;
    dat = (xtradata *)data;
    ap = dat->ap;
    //pzero    = 1.0; //Pzero*dat->Nnonzero[ap]/(dat->Nzero[ap]+0.01);
    
    distance_squared(d2,p,data); 
    for(i=j=0; i<n; ++i){
        jac[j++] = 0.0;
        jac[j++] = 0.0;
        
        if(sign(dat->z[i][ap])==1){
            cte = 2*(p[1]/d2[i]);
        }else{
            zp  = p[0]-p[1]*log(d2[i]);
            sigmoid_zp = sigmoid(zp);
            cte = 2*(p[1]/d2[2])*sigmoid_zp*(sigmoidk*(1-sigmoid_zp)*(zp-dat->z[i][ap])+1);
        }
        jac[j++] = cte*(dat->x[i]-p[2]);
        jac[j++] = cte*(dat->y[i]-p[3]);
    }
    PyMem_Free(d2);
}

void jacmodelfunp01(double *p, double *jac, int m, int n, void *data){   
/* Jacobian of modelfun() with first and second parameters fixed */
    register int i, j;
    double *d2 = (double *) PyMem_Malloc(sizeof(double)*n);
    double cte, zp, sigmoid_zp, ld2;
    int ap;

    xtradata *dat;
    dat = (xtradata *)data;
    ap = dat->ap;
    //pzero    = 1.0; //Pzero*dat->Nnonzero[ap]/(dat->Nzero[ap]+0.01);
    
    distance_squared(d2,p,data); 
    for(i=j=0; i<n; ++i){
        ld2 = log(d2[i]);
        if(sign(dat->z[i][ap])==1){
            cte = 1.0;
        }else{
            zp  = p[0]-p[1]*ld2;
            sigmoid_zp = sigmoid(zp);
            cte = sigmoid_zp*(sigmoidk*(1-sigmoid_zp)*(zp-dat->z[i][ap])+1);
        }
        jac[j++] = cte;
        jac[j++] = -cte*ld2;
        jac[j++] = 0.0;
        jac[j++] = 0.0;
        
    }
    PyMem_Free(d2);
}

void init_ap(double *p, void *data){
    double temp,p0init,p1init;
    double xmax=0,ymax=0,zmax=0,temax=0;
    double xavg=0,yavg=0,teavg=0;
    double xzc=0,yzc=0,zsum=0;
    register int i, j;
    int nzeros=0;
    int ap,n,m;

    xtradata *dat;
    dat = (xtradata *)data;
    ap = dat->ap;
    m  = dat->m;
    n  = dat->n;

    double e[n];

    p0init = p[0]; p1init=p[1];

    for(i=0; i<n; ++i){
        if(dat->z[i][ap]>0){
            zmax  = max(zmax,dat->z[i][ap]);
            zsum += dat->z[i][ap];
            xavg += dat->z[i][ap]*dat->x[i];
            yavg += dat->z[i][ap]*dat->y[i];
        }else{
            nzeros += 1;            
            xzc  += dat->x[i];
            yzc  += dat->y[i];
        }
    }
    
    xavg = xavg/zsum;
    yavg = yavg/zsum;
    if(debug){printf("Weighted xy:              \t %6.2f %6.2f\n", xavg, yavg);}      

    xzc  = xzc/nzeros;
    yzc  = yzc/nzeros;
    if(debug){printf("Zeros c:                  \t %6.2f %6.2f\n", xzc,  yzc);}      
    
    /* Distance from XYmax to XYzeros */
    xzc = xavg-xzc;
    yzc = yavg-yzc;
    /* Unitary vector*/
    temp  = sqrt(xzc*xzc+yzc*yzc);
    xzc = xzc/(temp+0.1);
    yzc = yzc/(temp+0.1);

    if(debug){printf("dXY to zeros c:           \t %6.2f %6.2f\n", xzc, yzc);}      
    temp  = (50-50*zmax);
    xmax = xavg+temp*xzc; //pushes p[2] away from center of zeros
    ymax = yavg+temp*yzc; //pushes p[3] away from center of zeros

    /* Try two optiosn, xmax far from xzeros, xavg */
    // Option 1    
    p[2] = xmax;  p[3] = ymax;    
    //solve_linear(p,data);
    //p[0] = min(p[0],1.);      p[0] = max(p[0],0.2);
    //p[1] = min(p[1],0.15);    p[1] = max(p[1],0.075);
    modelfun(p,e,m,n,data);
    temax = 0.0;
    for(i=0; i<n; i++){temax += e[i]*e[i];} //temax = N*sigma^2
    if(debug){printf("Shifted Max xy [sum of squares] \t %6.2f %6.2f [%12.9f]\n", xmax, ymax, temax);}    

    // Option 2
    p[0] = p0init; p[1] = p1init;    
    p[2] = xavg;   p[3] = yavg;
    //solve_linear(p,data);
    //p[0] = min(p[0],1.);      p[0] = max(p[0],0.2);
    //p[1] = min(p[1],0.15);    p[1] = max(p[1],0.075);
    modelfun(p,e,m,n,data);
    teavg = 0.0;
    for(i=0; i<n; i++){teavg += e[i]*e[i];}
    if(debug){printf("Weighted avg xy [sum of squares] \t %6.2f %6.2f [%12.9f]\n", xavg, yavg, teavg);}      

    if(teavg>temax){p[2] = xmax; p[3] = ymax;}
    if(debug){printf("Selected xy              \t %6.2f %6.2f\n", p[2], p[3]);}      

    p[0]  = p0init; p[1] = p1init;    

    if(debug){
        double jac[n*m];
        teavg = 0;        
        jacmodelfunp01(p,jac,m,n,data);
        teavg = 0;
        temax = 0;
        for(i=j=0; i<n; i++){
            teavg += pow(jac[j++],2);
            temax += pow(jac[j++],2);
            j++;
            j++;            
        }
        printf("Jac_p0 (Sum of squares) %10.7f\n",teavg);
        printf("Jac_p1 (Sum of squares) %10.7f\n",temax);
        teavg = 0;        
        jacmodelfunp23(p,jac,m,n,data);
        teavg = 0;
        temax = 0;
        for(i=j=0; i<n; i++){
            j++;
            j++;            
            teavg += pow(jac[j++],2);
            temax += pow(jac[j++],2);
        }
        printf("Jac_xap (Sum of squares) %10.7f\n",teavg);
        printf("Jac_yap (Sum of squares) %10.7f\n",temax);
    }
    //PyMem_Free(e);
}

void solve_linear(double *p, void *data){
    register int i;
    int N=0;
    double temp_d, temp_cte;
    double Sd=0, Sd2=0, St=0, Sdt=0;
    double dp0, dp1;
    int ap,n;

    xtradata *dat;
    dat = (xtradata *)data;
    ap = dat->ap;
    n  = dat->n;

    double *d2 = (double *) PyMem_Malloc(sizeof(double)*n);
    distance_squared(d2,p,data); 
    
    for(i=0; i<n; ++i){
        if(dat->z[i][ap]!=0){ // only nonzero values used
            temp_d = log(d2[i]);
            Sd  += temp_d;
            Sd2 += temp_d*temp_d;
            St  += dat->z[i][ap];
            Sdt += temp_d*dat->z[i][ap];
            N   += 1;
        }
    }

    temp_cte = 1/(N*Sd2-Sd*Sd);
    dp0 = temp_cte*(Sd2*St-Sd*Sdt)-p[0];
    dp1 = temp_cte*(Sd*St-N*Sdt)-p[1];
    // bound delta
    p[0] = p[0]+clip(dp0,-0.1,0.1); 
    p[1] = p[1]+clip(dp1,-0.01,0.01);
    // bound p0, p1
    p[0] = clip(p[0],0.2,1.0); 
    p[1] = clip(p[1],0.075,0.15);

    PyMem_Free(d2);
}

int load_variables(xtradata *data, PyObject *args){
    struct timeval t1, t2;
    double elapsedTime;

    gettimeofday(&t1, NULL);
       
    int i=0, j=0;
    int n, nap, Nnonzero, Nzero; // n is number of pointss, nap is number of access points
    PyObject *objx, *objy, *objz;

    // retrieve arguments from python
    if (!PyArg_ParseTuple(args, "OOO", &objx, &objy, &objz)) 
        return -1; //error not object

    // create iterators from objects
    PyObject *iterx = PyObject_GetIter(objx);
    if (!iterx)
        return -2; //error not object

    PyObject *itery = PyObject_GetIter(objy);
    if (!itery)
        return -2; //error not object

    PyObject *iterz = PyObject_GetIter(objz);
    if (!iterz)
        return -2; //error not object

    // load values to struct
    i = 0;
    if(debug){printf("Data X\n");}
    while(1) {
        PyObject *nextx = PyIter_Next(iterx);
        if (!nextx){break;} //end of iterator
        if (!PyFloat_Check(nextx)){return -3;} //error not object
        data->x[i] = (double) PyFloat_AsDouble(nextx);
        if(debug){printf("%f\n",data->x[i-1]);}
        i += 1;
    }
    n = i;
    data->n = n;

    i = 0;
    if(debug){printf("Data Y\n");}
    while(1) {
        PyObject *nexty = PyIter_Next(itery);
        if (!nexty){break;} //end of iterator
        if (!PyFloat_Check(nexty)){return -4;} //error not object
        data->y[i] = (double) PyFloat_AsDouble(nexty);
        if(debug){printf("%f\n",data->y[i]);}
        i += 1;
    }

    if(i!=n){return -5;} //error size of x and y do not match

    i = 0;
    if(debug){printf("Data Z\n");}
    while(1) {
        PyObject *nextz = PyIter_Next(iterz);
        if (!nextz){break;} //end of iterator
        if (!PyFloat_Check(nextz)){return -4;} //error not object
        data->z[i][j] = (double) PyFloat_AsDouble(nextz);
        if(debug){printf("%f\n",data->z[i][j]);}
        i ++;
        if (i==n){
            i = 0;            
            j++;
        }
    }
    nap = j;
    data->nap = nap;
    if(debug){printf("Number of access points: %d\n",nap);}
    if(i!=0){return -5;} //error size of x and z do not match

    //Compute Nnonzero and Nzero
    for(j=0; j<nap; ++j){
        Nnonzero=0; Nzero=0;
        for(i=0; i<n; ++i){
            if(data->z[i][j]>0){Nnonzero+=1;}
            else{Nzero+=1;}
        }
        data->Nnonzero[j] = Nnonzero;
        data->Nzero[j]    = Nzero;
        if(debug){printf("%4d: Nonzero %5d \tZero %5d\n",j,Nnonzero,Nzero);}
    }




    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Transfer data time %.7g ms.\n",elapsedTime);

    /* Computing x2y2 */
    for(i=0; i<n; ++i){
        data->x2y2[i] = pow(data->x[i],2) + pow(data->y[i],2);
    }

    return 0;
}



void irls(double *p, int m, int n, void *data){
    /* Levmar function*/
    int ret;

    double coarse_opts[LM_OPTS_SZ], info[LM_INFO_SZ];
    coarse_opts[0]=LM_INIT_MU; coarse_opts[1]=1E-10; coarse_opts[2]=1E-10; coarse_opts[3]=1E-10;

    /* optimization control parameters; passing to levmar NULL instead of opts reverts to defaults 
    double opts[LM_OPTS_SZ];
    opts[0]=LM_INIT_MU; opts[1]=1E-15; opts[2]=1E-15; opts[3]=1E-20;
    opts[4]=LM_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used 
    */

    // Optimization with p[0], p[1] fixed
    ret=dlevmar_der(modelfun, jacmodelfunp23, p, NULL, m, n, 200, coarse_opts, info, NULL, NULL, (void *)data); 
    if(ret!=-1){
        if(debug){
            printf("  [%.9f] -> [%.9f] Aftern Levenberg-Marquardt (%g iter, reason %g)\n",info[0],info[1],info[5],info[6]);
            printf("\t%.3f %.3f %.3f %.3f\n", p[0], p[1], p[2], p[3]);      
        }
    }
    
    // Optimization with p[2], p[3] fixed
    ret=dlevmar_der(modelfun, jacmodelfunp01, p, NULL, m, n, 200, coarse_opts, info, NULL, NULL, (void *)data); 
    if(ret!=-1){
        if(debug){
            printf("  [%.9f] -> [%.9f] Aftern Levenberg-Marquardt (%g iter, reason %g)\n",info[0],info[1],info[5],info[6]);
            printf("\t%.3f %.3f %.3f %.3f\n", p[0], p[1], p[2], p[3]);      
        }
    }

}

/* Python data input wrapper */ 
static PyObject *levmar_optimization(PyObject *self, PyObject *args){
    struct timeval t1, t2, t3, t4;
    double elapsedTime;

    int n,nap,load,convergence;
    int m=4; //m = number of parameters
    int i=0, j=0, iter=0;
    double p[m];            // single ap param array
    double p0_prev, p1_prev;

    if(debug)
        f = fopen("/home/renato/Desktop/parameters.txt", "w");
        f2= fopen("/home/renato/Desktop/results.txt","w");

    /* Load data from python */
    xtradata *data = (xtradata *) PyMem_Malloc(sizeof(xtradata));
    load = load_variables(data, args);
    if(load<0){ printf("Loading failed %d", load); return Py_BuildValue("n", load);}
    data->m = m; //set number of parameters in data struct
    
    /* Get parameters from loaded data */
    nap = data->nap; 
    n   = data->n;

    double p_all[m*nap];    // parameter array

    /* Main Loop */
    gettimeofday(&t1, NULL);
    for (i=0; i<nap; i++){
        convergence=0;
        p[0]=.6; p[1]=0.1;

        data->ap = i;

        init_ap(p,(void *)data);
        p0_prev = p[0];
        p1_prev = p[1];

        iter = 0;
        gettimeofday(&t2, NULL);
        if(debug){printf("For access point %d\n", i);}
        while(convergence==0){
            iter++;            
            
            //two step optimization
            irls(p,m,n,(void *)data);
            //solve_linear(p,data); if(debug){printf("\t%.3f %.3f %.3f %.3f (After reweighting)\n", p[0], p[1], p[2], p[3]);}
            //check convergence
            if (fabs(p[0]-p0_prev)<.05 && fabs(p[1]-p1_prev)<.005){
                convergence=1;
                if(debug){printf("\t%.3f %.3f %.3f %.3f (At convergence)\n", p[0], p[1], p[2], p[3]);}
                gettimeofday(&t3, NULL);
                elapsedTime = (t3.tv_sec - t2.tv_sec) * 1000.0;      // sec to ms
                elapsedTime += (t3.tv_usec - t2.tv_usec) / 1000.0;   // us to ms
                if(debug){fprintf(f2,"%6d: %6d iter  %9.3f\n",i,iter,elapsedTime);}
            }
            p0_prev = p[0];     p1_prev = p[1];
            if(debug){fprintf(f, "%.3f %.3f %.3f %.3f %.3f \n", p[0], p[1], p[2], p[3], 0.);}
        }
        for (j=0;j<m;j++){p_all[i*m+j] = p[j];}

        gettimeofday(&t4, NULL);
        elapsedTime = (t4.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t4.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        printf("Total time  %8.3f\n",elapsedTime);
    }
    
    PyObject *result;
    PyObject *value;

    result = PyTuple_New(m*nap);
    if (!result)
        return Py_BuildValue("n",6); //PyTuple_New error

    for (i = 0; i < m*nap; i++) {
        value = PyFloat_FromDouble(p_all[i]);
        if (!value){
            Py_DECREF(result);
            return  Py_BuildValue("n",7); //Py_Decref error
        }
        PyTuple_SetItem(result, i, value);
    }
    if(debug)
        fclose(f);
        fclose(f2);

    /* Freeing dynamic memory */
    PyMem_Free(data);

    /* Return parameter array */
    return result;
}



/* Python c api */
/* Method table */
static PyMethodDef LevmarMethods[] = {
    {"optimize",  levmar_optimization, METH_VARARGS, "Optimize parameters using levmar"},
    {NULL, NULL, 0, NULL}        // Sentinel 
};

/* Initialization function */
PyMODINIT_FUNC initlevmar(void){
    (void) Py_InitModule("levmar", LevmarMethods);
}

/* main */
int main(int argc, char *argv[]){
    Py_SetProgramName(argv[0]);     // Pass argv[0] to the Python interpreter
    Py_Initialize();    // Initialize Python interpreter
    initlevmar();       // Add static module
    Py_Exit(0);         // Close Python interpreter
    return 1;   
}
