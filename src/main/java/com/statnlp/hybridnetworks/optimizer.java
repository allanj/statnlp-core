package com.statnlp.hybridnetworks;

import com.statnlp.commons.ml.opt.LBFGS.ExceptionWithIflag;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.commons.ml.opt.Optimizer;
import com.statnlp.commons.ml.opt.LBFGSOptimizer;

import java.util.Arrays;

import com.statnlp.commons.ml.opt.LBFGS;
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
		System.load("/home/roozbeh/Documents/statnlp-core_2/target/optimizer.so"); 
	 }


	 
	 public native void initialize_weights(double[] weights);
	 public native void process();
	 public int N;
	 public double f;
	 public long wp;
	 public long gp;
	 public int m = 4;
	 public double epsilon = 10e-10;


	 static int _n;
	 static int _m = 4;
	 static double _f;
	 static double[] _diag;
	 static boolean _diagco = false;
	 static int[] _iprint = {0,0};
	 static int[] _iflag = {0};
	 static double _eps = 10e-10;
	 static double _xtol = 10e-16;
	 static LBFGSOptimizer opt_;
	 static NetworkModel g_net_model;
	 static double[] _weights;
	 static double zeros[]; 
	 

	 private static boolean updates() throws InterruptedException{
		g_net_model.done = g_net_model._fm._param_g.updateDiscriminative_post(g_net_model.done);
			if (!g_net_model.post_optimization())
				return false;
			
				g_net_model.pre_optimization();
			//done = true;
			g_net_model._fm._param_g.updateDiscriminative_pre();        
			g_net_model.done = false;
			return true;
	 }

	 public static void optimization_loop(NetworkModel net_model) throws InterruptedException{

		g_net_model =  net_model;
		optimizer opt = new optimizer();
		opt_ = (LBFGSOptimizer)net_model._fm._param_g._opt;
		opt.N = opt_._n;
		opt.initialize_weights(net_model._fm._param_g._weights);
		zeros = new double[opt.N];
		Arrays.fill(zeros, 0);

		
		opt.process();
		/*for(int it = 0; it<=net_model.max_iterations; it++){
			
				try{
					LBFGS.lbfgs(opt.N, _m,  opt.w, opt_._f, opt_._g, _diagco, opt_._diag, _iprint, _eps, _xtol, opt_._iflag);
				} catch(ExceptionWithIflag e){
					throw new NetworkException("Exception with Iflag:"+e.getMessage());
				}
		
			System.out.println("x:"+opt.w[1]);				
				
			if (!updates())
				break;
			
			System.out.println("g:"+opt_._g[1]);
			System.out.println("f:"+opt_._f);		
			System.out.println("\n");	

			}*/

	}
	 

	 @SuppressWarnings("restriction")
	public static void set(long p, int i, double d)
	  {
	    unsafe.putDouble(p+8*i,d);
	  }

	  @SuppressWarnings("restriction")
	public static double get(long p, int i)
	  {
	    return unsafe.getDouble(p+8*i);
	  }

	 private void evaluate() throws InterruptedException {
		//System.arraycopy(get_weights(), 0, _weights, 0, _weights.length);
		/*for(int i = 0; i<N; i++){
			g_net_model._fm._param_g._weights[i] = get(wp,i);
		}*/
		
		System.out.println("!x:"+opt_._x[1]);
		g_net_model._fm._param_g.gp = gp;

		if(!updates())
		{
			System.out.println("break");
		//	opt_._g = zeros;
		}
		//System.out.println("g:"+g_net_model._fm._param_g._counts[0]);
		System.out.println("f:"+opt_._f);	
		System.out.println("\n");		
		f = opt_._f;

		
	}

	 public  void main1(String args[]) {
	 	optimizer opt = new optimizer();
		opt.N = 6;

		opt.process();
		System.out.println("\n");
		System.out.println(opt.N);
	}
}

