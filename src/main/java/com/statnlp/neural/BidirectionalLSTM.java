package com.statnlp.neural;

import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

import scala.util.Random;
import th4j.Tensor.DoubleTensor;

import com.naef.jnlua.LuaState;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;
import com.sun.jna.Library;
import com.sun.jna.Native;

public class BidirectionalLSTM extends NeuralNetworkFeatureValueProvider {

	/**
	 * A LuaState instance for loading Lua scripts
	 */
	private LuaState L;

	/**
	 * Corresponding Torch tensors for params and gradParams
	 */
	private DoubleTensor paramsTensor, gradParamsTensor;
	
	/**
	 * Corresponding Torch tensors for output and gradOutput
	 */
	private DoubleTensor outputTensorBuffer, gradOutputTensorBuffer;
	
	/**
	 * CRF weight and gradient matrices.
	 * Shape: (numLabels x embeddingDimension)
	 */
	private SimpleMatrix weightMatrix, gradWeightMatrix;
	
	/**
	 * Neural network output and gradient matrices.
	 * Shape: (vocabSize x embeddingDimension)
	 */
	private SimpleMatrix outputMatrix, gradOutputMatrix;
	
	/**
	 * Result of outputMatrix * weightMatrix^T
	 * Shape: (vocabSize x numLabels)
	 */
	private SimpleMatrix forwardMatrix;
	
	/**
	 * The transposed weight matrix that will be multiplied with outputMatrix
	 */
	private SimpleMatrix weightMatrix_tran;
	
	/**
	 * Accumulated counts for CRF weights and neural network output.
	 * Count is typically computed by inside-outside.
	 */
	private SimpleMatrix countWeightMatrix, countOutputMatrix;
	
	/**
	 * Number of unique input sentences
	 */
	private int numSent;
	
	/**
	 * Maximum sentence length
	 */
	private int maxSentLen;
	
	/**
	 * Number of hidden units and layers
	 */
	private int hiddenSize;
	
	/**
	 * Whether CRF optimizes this neural network,
	 * same as defined by NetworkConfig.OPTIMIZE_NEURAL
	 */
	private boolean optimizeNeural;

	/**
	 * Map input sentence to index
	 */
	private TObjectIntHashMap<String> sentence2id;

	public BidirectionalLSTM(HashMap<String, Object> config, int numLabels) {
		super(config, numLabels);
		this.optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		this.sentence2id = new TObjectIntHashMap<String>();
		
		configureJNLua();
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
		Native.loadLibrary("libjnlua5.1.jnilib", Library.class);
		
		this.L = new LuaState();
		this.L.openLibs();
		
		try {
			this.L.load(Files.newInputStream(Paths.get("nn-crf-interface/neural_server/NetworkInterface.lua")),"NetworkInterface.lua");
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.L.call(0,0);
	}
	
	@Override
	public void initialize() {
		if (isTraining) {
			this.hiddenSize = (int) config.get("hiddenSize") * 2; // since bidirectional
		}
		makeInput();
		int vocabSize = numSent*maxSentLen;
				
		Random rng = null;
		if (isTraining) {
			rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			this.weightMatrix = new SimpleMatrix(numLabels, hiddenSize);
			this.weightMatrix_tran = new SimpleMatrix(hiddenSize, numLabels);
			this.gradWeightMatrix = new SimpleMatrix(numLabels, hiddenSize);
			DMatrixRMaj weightDMatrix = this.weightMatrix.getMatrix();
			this.weights = weightDMatrix.data;
			
			// Initialize weight matrix
			for(int i = 0; i < weights.length; i++) {
				weights[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
					NetworkConfig.FEATURE_INIT_WEIGHT;
			}
			DMatrixRMaj gradWeightDMatrix = this.gradWeightMatrix.getMatrix();
			this.gradWeights = gradWeightDMatrix.data;
			
			// Backward matrices
	        this.gradOutputMatrix = new SimpleMatrix(vocabSize, hiddenSize);
	        DMatrixRMaj gradOutputDMatrix = this.gradOutputMatrix.getMatrix();
	        this.gradOutput = gradOutputDMatrix.data;
	        this.countOutputMatrix = new SimpleMatrix(vocabSize, numLabels);
	        this.countWeightMatrix = new SimpleMatrix(numLabels, vocabSize);
	        
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(vocabSize, hiddenSize);
	        this.gradOutputTensorBuffer = new DoubleTensor(vocabSize, hiddenSize);
		}
		
		// Forward matrices
        this.outputMatrix = new SimpleMatrix(vocabSize, hiddenSize);
        this.forwardMatrix = new SimpleMatrix(vocabSize, numLabels);
		
		config.put("isTraining", isTraining);
        
        Object[] args = new Object[3];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.gradOutputTensorBuffer;
        Class<?>[] retTypes;
        if (optimizeNeural && isTraining) {
        	retTypes = new Class[]{DoubleTensor.class,DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "initialize", args, retTypes);
        
		if(optimizeNeural && isTraining) {
			this.paramsTensor = (DoubleTensor) outputs[0];
			this.gradParamsTensor = (DoubleTensor) outputs[1];
			if (this.paramsTensor.nElement() > 0) {
				this.params = getArray(this.paramsTensor);
				for(int i = 0; i < this.params.length; i++) {
					this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
				this.gradParams = getArray(this.gradParamsTensor);
			}
		}
	}
	
	public void makeInput() {
		Set<String> sentenceSet = new HashSet<String>();
		for (Object obj : input2id.keySet()) {
			@SuppressWarnings("unchecked")
			SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) obj;
			String sent = sentAndPos.getKey();
			sentenceSet.add(sent);
			int sentLen = sent.split(" ").length;
			if (sentLen > this.maxSentLen) {
				this.maxSentLen = sentLen;
			}
		}
		List<String> sentences = new ArrayList<String>(sentenceSet);
		for (int i = 0; i < sentences.size(); i++) {
			String sent = sentences.get(i);
			sentence2id.put(sent, i);
		}
		config.put("sentences", sentences);
		this.numSent = sentences.size();
		System.out.println("maxLen="+maxSentLen);
		System.out.println("#sent="+numSent);
	}

	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			@SuppressWarnings("unchecked")
			SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) input;
			int sentID = sentence2id.get(sentAndPos.getKey());
			int row = sentAndPos.getValue()*numSent+sentID; 
			val = forwardMatrix.get(row, outputLabel);
		}
		return val;
	}
	
	@Override
	public void forward() {
		if (optimizeNeural) { // update with new params
			if (getParamSize() > 0) {
				this.paramsTensor.storage().copy(this.params); // we can do this because params is contiguous
			}
		}
		
		Object[] args = new Object[]{isTraining};
		Class<?>[] retTypes = new Class[]{DoubleTensor.class};
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		
		// output is a tensor of size (maxLen x numSent) x hiddenSize
		output = getArray(outputTensorBuffer);
		DMatrixRMaj outputData = this.outputMatrix.getMatrix();
		outputData.data = output;
		
		CommonOps_DDRM.transpose(weightMatrix.getMatrix(), weightMatrix_tran.getMatrix());
		CommonOps_DDRM.mult(outputMatrix.getMatrix(), weightMatrix_tran.getMatrix(), forwardMatrix.getMatrix());
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			@SuppressWarnings("unchecked")
			SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) input;
			int sentID = sentence2id.get(sentAndPos.getKey());
			int row = sentAndPos.getValue()*numSent+sentID; 
			synchronized (countOutputMatrix) {
				double countOutput = countOutputMatrix.get(row, outputLabel);
				countOutputMatrix.set(row, outputLabel, countOutput-count);
			}
			synchronized (countWeightMatrix) {
				double countWeight = countWeightMatrix.get(outputLabel, row);
				countWeightMatrix.set(outputLabel, row, countWeight-count);
			}
		}
	}

	@Override
	public void backward() {
		// (vocabSize x numLabels) * (numLabels x hiddenSize)
		CommonOps_DDRM.mult(countOutputMatrix.getMatrix(), weightMatrix.getMatrix(), gradOutputMatrix.getMatrix());
		// (numLabels x vocabSize) * (vocabSize x hiddenSize)
		CommonOps_DDRM.mult(countWeightMatrix.getMatrix(), outputMatrix.getMatrix(), gradWeightMatrix.getMatrix());
		gradOutputTensorBuffer.storage().copy(gradOutput);
		
		Object[] args = new Object[]{};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(optimizeNeural && getParamSize() > 0) { // copy gradParams computed by Torch
			gradParams = getArray(this.gradParamsTensor);
		}
		
		addL2WeightsGrad();
		if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
			addL2ParamsGrad();
		}
		
		resetCount();
	}

	@Override
	public void save(String prefix) {
		// TODO Auto-generated method stub

	}

	@Override
	public void load(String prefix) {
		// TODO Auto-generated method stub

	}

	@Override
	public void cleanUp() {
		L.close();
	}

	public static HashMap<String, Object> createConfig(int hiddenSize, String optimizer) {
		HashMap<String, Object> config = new HashMap<String, Object>();
		config.put("class", "BidirectionalLSTM");
        config.put("hiddenSize", hiddenSize);
        config.put("optimizer", optimizer);
		return config;
	}
	
	public void resetCount() {
		countOutputMatrix.set(0.0);
		countWeightMatrix.set(0.0);
	}
	
	private double[] getArray(DoubleTensor t) {
		return t.storage().getRawData().getDoubleArray(0, (int) t.nElement());
	}
}
