package com.statnlp.hybridnetworks;
import java.util.*;
import sun.misc.Unsafe;
import java.lang.reflect.Field;


class optimizer {
  static Unsafe unsafe = null;
  //Native method declaration
   static {
        try {
            Field field = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            unsafe = (sun.misc.Unsafe) field.get(null);
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    String path =  optimizer.class.getProtectionDomain().getCodeSource().getLocation().getPath();
    System.load(path+"optimizer.so"); 
   }

   public native void initialize_weights(double[] weights);

   public native void process();
   public int N;
   public double f;
   public long wp;
   public long gp;
   public int m = 4;
   public double epsilon = 10e-10;



   void set(long p, int i, double d)
   {
    unsafe.putDouble(p+8*i,d);
   }

  double get(long p, int i)
  {
    return unsafe.getDouble(p+8*i);
  }



   private void evaluate() {
    f = 0.0;
    for (int i = 0;i < N;i += 2) {
    double t1 = 1.0 - get(wp,i);
    double t2 = 10.0 * (get(wp,i+1) - get(wp,i) * get(wp,i));
    set(gp,i+1,20.0 * t2); 
    set(gp,i, -2.0 * (get(wp,i) * get(gp,i+1) + t1));
    f += t1 * t1 + t2 * t2;
    }
  }

   public static void main(String args[]) {

    Unsafe unsafe = null;
        try {
            Field field = sun.misc.Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            unsafe = (sun.misc.Unsafe) field.get(null);
        } catch (Exception e) {
            throw new AssertionError(e);
        }

   	optimizer opt = new optimizer();
    opt.N = 6;
    opt.initialize_weights(new double[opt.N]);

    opt.process();
    System.out.println("\n");

    /*for (int i=0;i<opt.N;i++) {
      System.out.println("Value from c: " + unsafe.getDouble(opt.wp+8*i) + ", Mem: " + (long)(opt.wp+8*i));
    }*/
  }
}

