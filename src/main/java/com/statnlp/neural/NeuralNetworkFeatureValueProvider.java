package com.statnlp.neural;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;

import com.naef.jnlua.LuaState;
import com.statnlp.hybridnetworks.FeatureValueProvider;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;
import com.sun.jna.Library;
import com.sun.jna.Native;

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
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		super(numLabels);
		optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		config.put("optimizeNeural", optimizeNeural);
		//TODO: if optimize the neural in torch, need to pass the L2 val as well.
		this.configureJNLua();
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
			}
		}
		Object[] args = new Object[]{isTraining};
		Class<?>[] retTypes = new Class[]{DoubleTensor.class};
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		output = this.getArray(outputTensorBuffer, output);
		//output = outputTensorBuffer.storage().getRawData().getDoubleArray(0, (int)outputTensorBuffer.nElement());
		//getDoubleArray might be a bit faster, but requires more memory.
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int idx = this.input2Index(input) * this.numLabels + outputLabel;
			val = output[idx];
		}
		return val;
	}
	
	/**
	 * Neural network's backpropagation
	 */
	public void backward() {
		countOutputTensorBuffer.storage().copy(this.countOutput);
		
		Object[] args = new Object[]{};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(optimizeNeural && getParamSize() > 0) { // copy gradParams computed by Torch
			gradParams = this.getArray(this.gradParamsTensor, gradParams);
			if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
				addL2ParamsGrad();
			}
		}
		this.resetCountOutput();
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int idx = this.input2Index(input) * this.numLabels + outputLabel;
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
	 * TODO: should we clean up after training ? yes, if not decoding. No, if decode also, since decode will use it again.
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
