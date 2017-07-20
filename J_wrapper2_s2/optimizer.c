#include <jni.h>
#include "build/com_statnlp_hybridnetworks_optimizer.h"
#include <stdio.h>
#include "lbfgs.h"

jmethodID Evaluate;
JNIEnv *environment;
jobject *caller;
int N;
double* grad;
double* w;
jfieldID g_Id;
jfieldID f_Id;

static double evaluate(
    void *instance,
    const double *x,
    double *g,
    const int n,
    const double step
    )
{
	(*environment)->SetLongField(environment,caller, g_Id,(long)g);
    (*environment)->CallVoidMethod(environment, caller, Evaluate);  
    double f = (*environment)->GetDoubleField(environment,caller, f_Id);
    return f;
}




JNIEXPORT void JNICALL Java_com_statnlp_hybridnetworks_optimizer_initialize_1weights(JNIEnv *env, jobject obj, jdoubleArray jdA){
	 w = (*env)->GetDoubleArrayElements(env,jdA, NULL);
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



JNIEXPORT void JNICALL Java_com_statnlp_hybridnetworks_optimizer_process(JNIEnv *env, jobject jc) {

 jclass jcClass = (*env)->GetObjectClass(env,jc);
 environment = env;
 caller = jc;
 lbfgs_parameter_t param;

 param.m = (*env)->GetIntField(env,jc, (*env)->GetFieldID(env,jcClass, "m", "I"));
 param.epsilon = (*env)->GetIntField(env,jc, (*env)->GetFieldID(env,jcClass, "epsilon", "D"));


 jfieldID NId = (*env)->GetFieldID(env,jcClass, "N", "I");
 N = (*env)->GetIntField(env,jc, NId);
 
 jfieldID w_Id = (*env)->GetFieldID(env,jcClass, "wp", "J");
 long wp = (long) w;
 (*env)->SetLongField(env,jc, w_Id,wp);

 g_Id = (*env)->GetFieldID(env,jcClass, "gp", "J");
 f_Id = (*environment)->GetFieldID(environment,jcClass, "f", "D");

 Evaluate = (*env)->GetMethodID(env, jcClass, "evaluate", "()V");

  double f;

    if (w == NULL) {
        printf("ERROR: Weight array is not initialized.\n");
    }
 
    lbfgs_parameter_init(&param);

    int ret = lbfgs(N, w, &f, evaluate, progress, NULL, &param);

    printf("L-BFGS optimization terminated with status code = %d\n", ret);

}

