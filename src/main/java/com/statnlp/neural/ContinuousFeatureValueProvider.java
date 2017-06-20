package com.statnlp.neural;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;

import com.naef.jnlua.LuaState;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;
import com.sun.jna.Library;
import com.sun.jna.Native;

import scala.util.Random;
import th4j.Tensor.DoubleTensor;

public abstract class ContinuousFeatureValueProvider extends NeuralNetworkFeatureValueProvider {

	public static String LUA_VERSION = "5.2";
	private LuaState L;
	protected HashMap<String,Object> config;
	
	protected int numFeatureValues;
	
	public ContinuousFeatureValueProvider(int numLabels) {
		this(1, numLabels);
	}
	
	public ContinuousFeatureValueProvider(int numFeatureValues ,int numLabels) {
		super(numLabels);
		configureJNLua();
		this.numFeatureValues = numFeatureValues;
		config = new HashMap<>();
		config.put("class", "ContinuousFeature");
		config.put("numLabels", numLabels);
		config.put("numValues", numFeatureValues);
	}

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
	
	@Override
	public void initialize() {
		DoubleTensor inputs = makeInput();
		int inputSize = input2id.size();
		if (isTraining) {
			this.countOutput = new double[inputSize * this.numLabels];
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
	        this.countOutputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
		}
		// Forward matrices
		this.output = new double[inputSize * this.numLabels];
		
		config.put("isTraining", isTraining);
		
        Object[] args = new Object[4];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.countOutputTensorBuffer;
        args[3] = inputs;
        Class<?>[] retTypes;
        if (isTraining) {
        	retTypes = new Class[]{DoubleTensor.class, DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "initialize", args, retTypes);
        if(isTraining) {
        	this.paramsTensor = (DoubleTensor) outputs[0];
			this.gradParamsTensor = (DoubleTensor) outputs[1];
			Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			if (this.paramsTensor.nElement() > 0) {
				this.params = getArray(this.paramsTensor, this.params);
				for(int i = 0; i < this.params.length; i++) {
					this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
				this.gradParams = getArray(this.gradParamsTensor, this.gradParams);
			}
		}
	}
	
	@Override
	public void forward() {
		if (getParamSize() > 0) {
			this.paramsTensor.storage().copy(this.params); // we can do this because params is contiguous
		}
		Object[] args = new Object[]{isTraining};
		Class<?>[] retTypes = new Class[]{DoubleTensor.class};
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		output = getArray(outputTensorBuffer, output);
	}
	
	/**
	 * Fill the featureValue array using the input object
	 * @param input
	 * @return
	 */
	public abstract void getFeatureValue(Object input, double[] featureValue);
	
	public DoubleTensor makeInput() { 
		double[][] featureValues = new double[input2id.size()][this.numFeatureValues];
		for (Object input : input2id.keySet()) {
			this.getFeatureValue(input, featureValues[input2id.get(input)]);
			//input2value.put(input, featureValue);
		}
		DoubleTensor dt = new DoubleTensor(featureValues);
		return dt;
	}

	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int inputId = input2id.get(input);
			val = output[inputId * this.numLabels + outputLabel];
		}
		return val;
	}

	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int inputId = input2id.get(input);
			synchronized (countOutput) {
				int idx = inputId * this.numLabels + outputLabel;
				countOutput[idx] -= count;
			}
		}
	}
	
	@Override
	public void backward() {
		countOutputTensorBuffer.storage().copy(this.countOutput);
		
		Object[] args = new Object[]{};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(getParamSize() > 0) { // copy gradParams computed by Torch
			gradParams = getArray(this.gradParamsTensor, gradParams);
		}
		if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
			addL2ParamsGrad();
		}
		resetCount();
	}

	@Override
	public void save(String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, "save_model", new Object[]{prefix}, new Class[]{});
	}

	@Override
	public void load(String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, "load_model", new Object[]{prefix}, new Class[]{});
	}

	@Override
	public void cleanUp() {
		L.close();
	}

	public void resetCount() {
		Arrays.fill(countOutput, 0.0);
	}
	
	private double[] getArray(DoubleTensor t, double[] buf) {
		if (buf == null || buf.length != t.nElement()) {
			buf = new double[(int) t.nElement()];
        }
		t.storage().getRawData().read(0, buf, 0, (int) t.nElement());
		return buf;
	}

}
