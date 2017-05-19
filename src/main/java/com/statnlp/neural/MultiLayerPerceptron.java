package com.statnlp.neural;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import scala.collection.Iterator;
import th4j.Tensor.DoubleTensor;

import com.naef.jnlua.LuaState;
import com.statnlp.commons.types.Instance;
import com.statnlp.neural.util.LuaFunctionHelper;
import com.sun.jna.Library;
import com.sun.jna.Native;

/**
 * The class that serves as the interface to access the neural network backend.
 * This uses TH4J and JNLua to transfer the data between the JVM and the NN backend.
 */
public class MultiLayerPerceptron extends AbstractNetwork {
	private boolean DEBUG = false;
	
	// Torch NN server information
	private LuaState L;
	private DoubleTensor params, gradParams;
	private int numNetworks, totalOutputDim, vocabSize;
	private List<Integer> outputDimList;
	private DoubleTensor[] outputTensorBuffer;
	private DoubleTensor[] gradTensorBuffer;

	public MultiLayerPerceptron(boolean optimizeNeural) {
		super(optimizeNeural);
		
		configure();
		this.L = new LuaState();
		this.L.openLibs();
		
		try {
			this.L.load(Files.newInputStream(Paths.get("nn-crf-interface/neural_server/server-jni.lua")),"server-jni.lua");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		this.L.call(0,0);
	}
	
	private void configure() {
		System.setProperty("jna.library.path","./nativeLib");
		System.setProperty("java.library.path", "./nativeLib:" + System.getProperty("java.library.path"));
		Field fieldSysPath = null;
		try {
			fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
			fieldSysPath.setAccessible(true);
			fieldSysPath.set(null, null);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Native.loadLibrary("libjnlua5.1.jnilib", Library.class);
	}
	
	public double[] initNetwork(List<Integer> numInputList, List<Integer> inputDimList, List<String> wordList,
						   String lang, List<String> embeddingList, List<Integer> embSizeList,
						   List<Integer> outputDimList, List<List<Integer>> vocab) {
		Map<String, Object> config = new HashMap<String, Object>();
		config.put("numInputList", numInputList);
        config.put("inputDimList", inputDimList);
        config.put("wordList", wordList);
        config.put("lang", lang);
        config.put("embedding", embeddingList);
        config.put("embSizeList", embSizeList);
        config.put("outputDimList", outputDimList);
        config.put("numLayer", NeuralConfig.NUM_LAYER);
        config.put("hiddenSize", NeuralConfig.HIDDEN_SIZE);
        config.put("activation", NeuralConfig.ACTIVATION);
        config.put("dropout", NeuralConfig.DROPOUT);
        config.put("optimizer", NeuralConfig.OPTIMIZER);
        config.put("learningRate", NeuralConfig.LEARNING_RATE);
        config.put("fixEmbedding", NeuralConfig.FIX_EMBEDDING);
        config.put("useOutputBias", NeuralConfig.USE_OUTPUT_BIAS);
        config.put("vocab", vocab);
        
        this.vocabSize = vocab.size();
        this.outputDimList = outputDimList;
        this.numNetworks = outputDimList.size();
        for (int outputDim : outputDimList) {
        	this.totalOutputDim += outputDim;
        }
        
        // 2D buffer array to be used by backward()
        this.outputTensorBuffer = new DoubleTensor[this.numNetworks];
        this.gradTensorBuffer = new DoubleTensor[this.numNetworks];
        for (int i = 0; i < this.gradTensorBuffer.length; i++) {
        	int outputDim = outputDimList.get(i);
        	this.outputTensorBuffer[i] = new DoubleTensor(this.vocabSize, outputDim);
        	this.gradTensorBuffer[i] = new DoubleTensor(this.vocabSize, outputDim);
        }
        
        Object[] args = new Object[2*this.numNetworks+1];
        args[0] = config;
        for (int i = 0; i < 2*this.numNetworks; i++) {
        	if (i < this.numNetworks) {
        		args[i+1] = this.outputTensorBuffer[i];
        	} else {
        		args[i+1] = this.gradTensorBuffer[i-this.numNetworks];
        	}
        }
        Class<?>[] retTypes;
        if (optimizeNeural) {
        	retTypes = new Class[]{DoubleTensor.class,DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "init_MLP", args, retTypes);
        
		double[] nnInternalWeights = null;
		if(optimizeNeural) {
			this.params = (DoubleTensor) outputs[0];
			long size = this.params.nElement();
			nnInternalWeights = new double[(int) size];
			Iterator<Object> iter = this.params.iterator();
			int cnt = 0;
			while (iter.hasNext()) {
				nnInternalWeights[cnt++] = (double) iter.next();
			}
			this.gradParams = (DoubleTensor) outputs[1];
		}
		return nnInternalWeights;
	}
	
	public void forwardNetwork(boolean training) {
		if (optimizeNeural) { // update with new params
			double[] nnInternalWeights = controller.getInternalNeuralWeights();
			this.params.storage().copy(nnInternalWeights); // we can do this because params is contiguous
		}
		
		Object[] args = new Object[]{training};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "fwd_MLP", args, retTypes);
		
		// copy forward result
		double[] nnExternalWeights = new double[this.vocabSize*this.totalOutputDim];
		int cnt = 0;
		for (int i = 0; i < this.numNetworks; i++) {
			int len = this.vocabSize*this.outputDimList.get(i);
			DoubleTensor t = outputTensorBuffer[i];
			double[] tmp = t.storage().getRawData().getDoubleArray(0, len);
			System.arraycopy(tmp, 0, nnExternalWeights, cnt, len);
			cnt += len;
		}
		controller.updateExternalNeuralWeights(nnExternalWeights);
	}
	
	public void backwardNetwork() {
		double[] grad = controller.getExternalNeuralGradients();
		
		Object[] args = new Object[0];
		int cnt = 0;
		for (int i = 0; i < this.numNetworks; i++) {
			int outputDim = this.outputDimList.get(i);
			int len = this.vocabSize*outputDim;
			double[] tmp = Arrays.copyOfRange(grad, cnt, cnt+len);
			gradTensorBuffer[i].storage().copy(tmp);
		}
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "bwd_MLP", args, retTypes);
		
		if(optimizeNeural) { // copy gradParams computed by Torch
			controller.setInternalNeuralGradients(this.gradParams.storage().getRawData().getDoubleArray(0, (int) this.gradParams.nElement()));
		}
	}
	
	public void saveNetwork(String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, "save_model", new Object[]{prefix}, new Class[]{});
	}
	
	public void loadNetwork(String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, "load_model", new Object[]{prefix}, new Class[]{});
	}
	
	public void cleanUp() {
		L.close();
	}
	
	public static void main(String[] args) {
		MultiLayerPerceptron nn = new MultiLayerPerceptron(true);
		List<Integer> numInputList = Arrays.asList(1);
		List<Integer> inputDimList = Arrays.asList(5);
		List<String> wordList = Arrays.asList("a","b","c","d","e");
		List<String> embeddingList = Arrays.asList("none");
		List<Integer> embSizeList = Arrays.asList(3);
		List<Integer> outputDimList = Arrays.asList(2);
		List<List<Integer>> vocab = new ArrayList<List<Integer>>();
		for (int i = 1; i <= 5; i++) {
			vocab.add(Arrays.asList(i));
		}
		try {
			NeuralConfigReader.readConfig("nn-crf-interface/neural_server/neural.basic.config");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		nn.initNetwork(numInputList, inputDimList, wordList, "en", embeddingList, embSizeList, outputDimList, vocab);
//		nn.forwardNetwork(true);
//		nn.backwardNetwork();
	}

	@Override
	public void setInput(Instance[] instances) {
		// TODO Auto-generated method stub
		
	}
}
