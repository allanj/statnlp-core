package com.statnlp.hybridnetworks;
import java.util.*;


class optimizer {
  //Native method declaration
   static {
    String path =  optimizer.class.getProtectionDomain().getCodeSource().getLocation().getPath();
    System.load(path+"optimizer.so"); 
   }

   public native void set_weights(double[] weights);
   public native double[] get_weights();
   public native void set_gradients(double[] weights);

   public native void process();
   public int N;
   public double f;


   private void evaluate() {
    double[] w = get_weights();
    double[] g = new double[N];
    f = 0.0;
    for (int i = 0;i < N;i += 2) {
    double t1 = 1.0 - w[i];
    double t2 = 10.0 * (w[i+1] - w[i] * w[i]);
    g[i+1] = 20.0 * t2;
    g[i] = -2.0 * (w[i] * g[i+1] + t1);
    f += t1 * t1 + t2 * t2;
    }
    set_gradients(g);
  }

   public static void main(String args[]) {
   	optimizer opt = new optimizer();
    opt.N = 6;
    opt.set_weights(new double[opt.N]);

    opt.process();
    System.out.println("\n");
    System.out.println(opt.N);
  }
}

