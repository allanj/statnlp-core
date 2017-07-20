#include <jni.h>
#include "build/NetworkModel.h"
#include <stdio.h>
#include "lbfgs.h"

jmethodID Evaluate;
JNIEnv *environment;
jobject *caller;
int N;
double* w;
double* grad;

static double evaluate(
    void *instance,
    const double *x,
    double *g,
    const int n,
    const double step
    )
{
    grad = g;
    (*environment)->CallVoidMethod(environment, caller, Evaluate);  

    jclass jcClass = (*environment)->GetObjectClass(environment,caller);    
    /*jfieldID g_Id = (*environment)->GetFieldID(environment,jcClass, "g", "[D");
 	jdoubleArray g_j = (*environment)->GetObjectField (environment,caller, g_Id);    
    (*environment)->GetDoubleArrayRegion(environment,g_j, 0,N,g);*/
    
    jfieldID f_Id = (*environment)->GetFieldID(environment,jcClass, "f", "D");
    double f = (*environment)->GetDoubleField(environment,caller, f_Id);
    return f;
}


JNIEXPORT void JNICALL Java_NetworkModel_set_1gradients(JNIEnv *env, jobject obj, jdoubleArray jdA){
    //jclass jcClass = (*environment)->GetObjectClass(environment,caller);    
    //jfieldID g_Id = (*environment)->GetFieldID(environment,jcClass, "g", "[D");
    //jdoubleArray g_j = (*environment)->GetObjectField (environment,caller, g_Id);    
    (*environment)->GetDoubleArrayRegion(environment,jdA, 0,N,grad);
}


JNIEXPORT void JNICALL Java_NetworkModel_set_1weights(JNIEnv *env, jobject obj, jdoubleArray jdA){
	 w = (*env)->GetDoubleArrayElements(env,jdA, NULL);
}

JNIEXPORT jdoubleArray JNICALL Java_NetworkModel_get_1weights(JNIEnv *env, jobject obj) {
     jdoubleArray w_j = (*environment)->NewDoubleArray(environment,N);
    (*environment)->SetDoubleArrayRegion(environment, w_j, 0, N, w);
    return w_j;
}

static int progress(
    void *instance,
    const double *w,
    const double *g,
    const double f,
    const double xnorm,
    const double gnorm,
    const double step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d:\n", k);
    printf("  f = %f, w[0] = %f, w[1] = %f\n", f, w[0], w[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}



JNIEXPORT void JNICALL Java_NetworkModel_optimize(JNIEnv *env, jobject jc) {

 jclass jcClass = (*env)->GetObjectClass(env,jc);
 environment = env;
 caller = jc;
 jfieldID NId = (*env)->GetFieldID(env,jcClass, "N", "I");
 N = (*env)->GetIntField(env,jc, NId);
 Evaluate = (*env)->GetMethodID(env, jcClass, "evaluate", "()V");


  int  ret = 0;

  double f;

  lbfgs_parameter_t param;

    if (w == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return 1;
    }


    lbfgs_parameter_init(&param);

    ret = lbfgs(N, w, &f, evaluate, progress, NULL, &param);

    printf("L-BFGS optimization terminated with status code = %d\n", ret);
    printf("  f = %f, w[0] = %f, w[1] = %f\n", f, w[0], w[1]);

    lbfgs_free(w);
}

