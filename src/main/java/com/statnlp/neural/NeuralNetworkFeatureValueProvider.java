package com.statnlp.neural;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

import com.naef.jnlua.LuaState;
import com.statnlp.hybridnetworks.FeatureValueProvider;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;
import com.sun.jna.Library;
import com.sun.jna.Native;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;
import th4j.Tensor.DoubleTensor;

public abstract class NeuralNetworkFeatureValueProvider extends FeatureValueProvider {
	
	public static String LUA_VERSION = "5.2";
	
	/**
	 * A LuaState instance for loading Lua scripts
	 */
	public LuaState L;

	protected HashMap<String,Object> config = new HashMap<>();;
	
	/**
	 * Corresponding Torch tensors for params and gradParams
	 */
	protected DoubleTensor paramsTensor, gradParamsTensor;
	
	/**
	 * Corresponding Torch tensors for output and gradOutput
	 */
	protected DoubleTensor outputTensorBuffer, countOutputTensorBuffer;
	
	public boolean optimizeNeural;
	
	/**
	 * The current batch FVP input ids.
	 */
	private TIntArrayList currBatchFVPInputIds;
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		super(numLabels);
		optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		currBatchFVPInputIds = new TIntArrayList();
		config.put("optimizeNeural", optimizeNeural);
		//TODO: if optimize the neural in torch, need to pass the L2 val as well.
		this.configureJNLua();
	}
	
	@Override
	public void initialize() {
		this.initializeNN();
		if (this.isTraining && NetworkConfig.USE_BATCH_TRAINING) {
			this.makeInstsId2FVPInputId();
		}
	}
	
	public abstract void initializeNN();
	
	/**
	 * Make sure the fvpInput2Id is now fixed.
	 */
	private void makeInstsId2FVPInputId() {
		for (int instanceId : this.instId2FVPInput.keys()) {
			Set<Object> fvpInputs = this.instId2FVPInput.get(instanceId);
			TIntList fvpInputIds = new TIntArrayList();
			for (Object fvpInput : fvpInputs) {
				fvpInputIds.add(fvpInput2id.get(fvpInput));
			}
			this.instId2FVPInputId.put(instanceId, fvpInputIds);
		}
		System.out.println("inst id to fvp input id: " + this.instId2FVPInputId.toString());
		this.instId2FVPInput = null;
	}
	
	@Override
	public void initializeScores() {
		forward();
	}
	
	/**
	 * Configure paths for JNLua and create a new LuaState instance
	 * for loading the backend Torch/Lua script
	 */
	private void configureJNLua() {
		System.setProperty("jna.library.path","./nativeLib");
		System.setProperty("java.library.path", "./nativeLib:" + System.getProperty("java.library.path"));
		Field fieldSysPath = null;
		try {
			fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
			fieldSysPath.setAccessible(true);
			fieldSysPath.set(null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
		String jnluaLib = null;
		if (LUA_VERSION.equals("5.2")) {
			jnluaLib = "libjnlua52";
		} else if (LUA_VERSION.equals("5.1")) {
			jnluaLib = "libjnlua5.1";
		}
		if (NetworkConfig.OS.equals("osx")) {
			jnluaLib += ".jnilib";
		} else if (NetworkConfig.OS.equals("linux")) {
			jnluaLib += ".so";
		}
		Native.loadLibrary(jnluaLib, Library.class);
		
		this.L = new LuaState();
		this.L.openLibs();
		
		try {
			this.L.load(Files.newInputStream(Paths.get("nn-crf-interface/neural_server/NetworkInterface.lua")),"NetworkInterface.lua","bt");
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.L.call(0,0);
	}
	
	/**
	 * Calculate the input position in the output/countOuput matrix position
	 * @return
	 */
	public abstract int input2Index(Object input);
	
	@Override
	public void update() {
		backward();
	}
	
	/**
	 * Neural network's forward
	 */
	public void forward() {
		if (optimizeNeural) { // update with new params
			if (getParamSize() > 0) {
				this.paramsTensor.storage().copy(this.params); // we can do this because params is contiguous
				//System.out.println("java side forward weights: " + this.params[0] + " " + this.params[1]);
			}
		}
		Object[] args = null; 
		if (isTraining && NetworkConfig.USE_BATCH_TRAINING) {
			TIntIterator iter = this.batchInstIds.iterator();
			TIntHashSet set = new TIntHashSet();
			while(iter.hasNext()) {
				int positiveInstId = iter.next();
				assert(positiveInstId > 0);
				//the following two might be the same.
				System.out.println("positive id: "+ positiveInstId);
				set.addAll(this.instId2FVPInputId.get(positiveInstId));
				int s1 = set.size();
				set.addAll(this.instId2FVPInputId.get(-positiveInstId));
				int s2 = set.size();
				if (s1 != s2) {
					throw new RuntimeException("the two input lists are not the same?");
				}
			}
			currBatchFVPInputIds = new TIntArrayList(set);
			args = new Object[]{isTraining, this.currBatchFVPInputIds};
		} else {
			args = new Object[]{isTraining};
		}
		Class<?>[] retTypes = new Class[]{DoubleTensor.class};
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		output = this.getArray(outputTensorBuffer, output);
		//System.out.println("java side forward output: " + this.output[0] + " " + this.output[1]);
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object edgeInput = getHyperEdgeInput(network, parent_k, children_k_index);
		if (edgeInput != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int idx = this.input2Index(edgeInput) * this.numLabels + outputLabel;
			val = output[idx];
		}
		return val;
	}
	
	/**
	 * Neural network's backpropagation
	 */
	public void backward() {
		countOutputTensorBuffer.storage().copy(this.countOutput);
		/****  //debug code can be removed
		double sumGradOutput = 0;
		for (int i = 0; i< countOutput.length; i++)
			sumGradOutput += this.countOutput[i];
		System.out.println("java side back sum gradOutput: " + sumGradOutput);
		System.out.println("java side back gradOutput: " + this.countOutput[0] + " " + this.countOutput[1]);
		***/
		Object[] args = new Object[]{};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(optimizeNeural && getParamSize() > 0) { // copy gradParams computed by Torch
			gradParams = this.getArray(this.gradParamsTensor, gradParams);
			/****  //debug code can be removed
			//double sum = 0;
			//for (int i = 0; i< gradParams.length; i++)
			//	sum += this.gradParams[i];
			//System.out.println("java side back gradPram: " + sum + "  param length:"+gradParams.length);
			***/
			if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
				addL2ParamsGrad();
			}
		}
		this.resetCountOutput();
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object edgeInput = getHyperEdgeInput(network, parent_k, children_k_index);
		if (edgeInput != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int idx = this.input2Index(edgeInput) * this.numLabels + outputLabel;
			synchronized (countOutput) {
				countOutput[idx] -= count;
			}
		}
	}
	
	public void resetCountOutput() {
		Arrays.fill(countOutput, 0.0);
	}
	
	/**
	 * Save the model by calling the specific function in Torch
	 * @param func : the function in torch
	 * @param prefix : model prefix
	 */
	public void save(String func, String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, func, new Object[]{prefix}, new Class[]{});
	}
	
	/**
	 * Save the trained model, implement the "save_model" method in torch
	 * @param prefix
	 */
	public void save(String prefix) {
		this.save("save_model", prefix);
	}
	
	/**
	 * Load a trained model, using the specific function in Torch
	 * @param func: the specific function for loading model
	 * @param prefix: model prefix.
	 */
	public void load(String func, String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, func, new Object[]{prefix}, new Class[]{});
	}
	
	/**
	 * Load a model from disk, implement the "load_model" method in torch
	 * @param prefix
	 */
	public void load(String prefix) {
		this.load("load_model", prefix);
	}
	
	@Override
	public void closeProvider() {
		this.cleanUp();
	}
	
	/**
	 * Clean up resources, currently, we clean up the resource after decoding
	 */
	public void cleanUp() {
		L.close();
	}
	
	/**
	 * Read a DoubleTensor to a buffer.
	 * @param t
	 * @param buf
	 * @return
	 */
	protected double[] getArray(DoubleTensor t, double[] buf) {
		if (buf == null || buf.length != t.nElement()) {
			buf = new double[(int) t.nElement()];
        }
		t.storage().getRawData().read(0, buf, 0, (int) t.nElement());
//		Iterator<Object> iter = t.iterator();
//		int ptr = 0;
//		while (iter.hasNext()) { // manual iteration like this is actually slow
//			buf[ptr++] = (double) iter.next(); 
//		}
		return buf;
	}
}
