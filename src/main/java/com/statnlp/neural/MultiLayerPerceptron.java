package com.statnlp.neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

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

/**
 * The class that serves as the interface to access the neural network backend.
 * This uses TH4J and JNLua to transfer the data between the JVM and the NN backend.
 */
public class MultiLayerPerceptron extends NeuralNetworkFeatureValueProvider {
	
	/**
	 * Special delimiters for the input.
	 * e.g., w1#IN#w2#IN#w3#OUT#t1#IN#t2#IN#t3
	 */
	public static final String IN_SEP = "#IN#";
	public static final String OUT_SEP = "#OUT#";
	
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
	 * Whether CRF optimizes this neural network,
	 * same as defined by NetworkConfig.OPTIMIZE_NEURAL
	 */
	private boolean optimizeNeural;
	
	/**
	 * List of word embedding to be used,
	 * e.g., "polyglot", "glove"
	 */
	private List<String> embeddingList;
	
	/**
	 * Number of unique inputs to this neural network 
	 */
	private int vocabSize;
	
	/**
	 * Number of hidden units and layers
	 */
	private int hiddenSize, numLayer;
	
	/**
	 * Total dimension of the input layers,
	 * i.e., sum of embedding size for each input type
	 */
	private int totalInputDim;
	
	/**
	 * Window size for each input type
	 */
	private List<Integer> numInputList;
	
	/**
	 * Number of unique tokens for each input type
	 */
	private List<Integer> inputDimList;
	
	/**
	 * List of token2idx mappings.
	 * One for each input type (word, tag, etc.)
	 */
	private List<HashMap<String,Integer>> token2idxList;

	public MultiLayerPerceptron(HashMap<String, Object> config, int numLabels) {
		super(config, numLabels);
		this.optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		
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
	
	@SuppressWarnings("unchecked")
	public void initialize() {
		if (isTraining) {
			this.embeddingList = (List<String>) config.get("embedding");
			this.hiddenSize = (int) config.get("hiddenSize");
			this.numLayer = (int) config.get("numLayer");
			this.numInputList = new ArrayList<Integer>();
			this.inputDimList = new ArrayList<Integer>();
			this.token2idxList = new ArrayList<HashMap<String,Integer>>();
		}
		makeVocab();
		
		Random rng = null;
		if (isTraining) {
			totalInputDim = 0;
			List<Integer> embSizeList = (List<Integer>) config.get("embSizeList");
			for (int i = 0; i < token2idxList.size(); i++) {
				if (embSizeList.get(i) != 0) {
					totalInputDim += embSizeList.get(i);
				} else {
					totalInputDim += token2idxList.get(i).size();
				}
			}
			
			rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			
			this.weightMatrix = new SimpleMatrix(numLabels, getHiddenSize());
			this.weightMatrix_tran = new SimpleMatrix(getHiddenSize(), numLabels);
			this.gradWeightMatrix = new SimpleMatrix(numLabels, getHiddenSize());
			DMatrixRMaj weightDMatrix = this.weightMatrix.getMatrix();
			this.weights = weightDMatrix.data;
			
			// Initialize weight matrix
			double stdv = 1.0/Math.sqrt(getHiddenSize());
			for(int i = 0; i < weights.length; i++) {
				if (NetworkConfig.NEURAL_RANDOM_TYPE.equals("xavier")) {
					weights[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? -stdv + 2 * stdv * rng.nextDouble() :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				} else {
					weights[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
			}
			DMatrixRMaj gradWeightDMatrix = this.gradWeightMatrix.getMatrix();
			this.gradWeights = gradWeightDMatrix.data;
			
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(getVocabSize(), getHiddenSize());
	        this.gradOutputTensorBuffer = new DoubleTensor(getVocabSize(), getHiddenSize());
	        
	        // Backward matrices
	        this.gradOutputMatrix = new SimpleMatrix(getVocabSize(), getHiddenSize());
	        DMatrixRMaj gradOutputDMatrix = this.gradOutputMatrix.getMatrix();
	        this.gradOutput = gradOutputDMatrix.data;
	        this.countOutputMatrix = new SimpleMatrix(getVocabSize(), numLabels);
	        this.countWeightMatrix = new SimpleMatrix(numLabels, getVocabSize());
		}
		
        // Forward matrices
        this.outputMatrix = new SimpleMatrix(getVocabSize(), getHiddenSize());
        this.forwardMatrix = new SimpleMatrix(getVocabSize(), numLabels);
        
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
				this.gradParams = getArray(this.gradParamsTensor);
			}
		}
	}
	
	/**
	 * Helper method to convert the input from String form
	 * (e.g., w1#IN#w2#IN#w3#OUT#t1#IN#t2#IN#t3) to integer indices
	 * and gather input information (numInputList, inputDimList).
	 */
	private void makeVocab() {
		List<String> wordList = new ArrayList<String>();
		List<List<Integer>> vocab = new ArrayList<List<Integer>>();
		boolean first = true;
		for (Object obj : input2id.keySet()) {
			String input = (String) obj;
			String[] inputPerType = input.split(OUT_SEP);
			List<Integer> entry = new ArrayList<Integer>();
			for (int i = 0; i < inputPerType.length; i++) {
				String[] tokens = inputPerType[i].split(IN_SEP);
				if (first && isTraining) {
					numInputList.add(tokens.length);
					inputDimList.add(0);
					token2idxList.add(new HashMap<String, Integer>());
				}
				HashMap<String,Integer> token2idx = token2idxList.get(i);
				for (int j = 0; j < tokens.length; j++) {
					String token = tokens[j];
					if (!token2idx.containsKey(token)) {
						inputDimList.set(i, inputDimList.get(i)+1);
						int idx = NetworkConfig.IS_INDEXED_NEURAL_FEATURES? Integer.parseInt(token):token2idx.size();
						token2idx.put(token, idx);
						if (embeddingList.get(i).equals("glove")
							|| embeddingList.get(i).equals("polyglot")) {
							wordList.add(token);
						}
					}
					int idx = token2idx.get(token);
					if (NetworkConfig.NEURAL_BACKEND.startsWith("torch")) { // 1-indexing
						idx++;
					}
					entry.add(idx);
				}
			}
			vocab.add(entry);
			first = false;
		}
		config.put("numInputList", numInputList);
        config.put("inputDimList", inputDimList);
        config.put("wordList", wordList);
        config.put("vocab", vocab);
        vocabSize = vocab.size();
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int id = input2id.get(input);
			val = forwardMatrix.get(id, outputLabel);
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
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		
		// copy forward result
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
			int id = input2id.get(input);
			synchronized (countOutputMatrix) {
				double countOutput = countOutputMatrix.get(id, outputLabel);
				countOutputMatrix.set(id, outputLabel, countOutput+count);
			}
			synchronized (countWeightMatrix) {
				double countWeight = countWeightMatrix.get(outputLabel, id);
				countWeightMatrix.set(outputLabel, id, countWeight-count);
			}
		}
	}
	
	@Override
	public void backward() {
		// inputSize x numLabel * numLabel x hiddenSize
		CommonOps_DDRM.mult(countOutputMatrix.getMatrix(), weightMatrix.getMatrix(), gradOutputMatrix.getMatrix());
		// numLabel x inputSize * inputSize x hiddenSize
		CommonOps_DDRM.mult(countWeightMatrix.getMatrix(), outputMatrix.getMatrix(), gradWeightMatrix.getMatrix());
		
//		double ds[] = new double[numLabels*vocabSize];
		double ds[] = new double[vocabSize*hiddenSize];
		int ptr = 0;
		for(int i = 0; i < vocabSize; i++) {
			for (int j = 0; j < hiddenSize; j++) {
				ds[ptr++]=outputMatrix.get(i,j);
			}
		}
		Arrays.sort(ds);
		ptr = 1;
		for(double d : ds) {
			System.out.println(ptr+":"+d);
			ptr++;
		}
		
		gradOutputTensorBuffer.storage().copy(gradOutput);
		
		Object[] args = new Object[0];
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
		LuaFunctionHelper.execLuaFunction(this.L, "save_model", new Object[]{prefix}, new Class[]{});
	}
	
	@Override
	public void load(String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, "load_model", new Object[]{prefix}, new Class[]{});
	}
	
	public void cleanUp() {
		L.close();
	}
	
	/**
	 * Convenience method to generate a config Map for this neural network
	 * @param lang
	 * @param embeddingList
	 * @param embSizeList
	 * @param numLayer
	 * @param hiddenSize
	 * @param activation
	 * @param dropout
	 * @param optimizer
	 * @param learningRate
	 * @param fixInputLayer
	 * @param useOutputBias
	 * @return
	 */
	public static HashMap<String, Object> createConfig(
			String lang, List<String> embeddingList, List<Integer> embSizeList,
			int numLayer, int hiddenSize, String activation,
			double dropout, String optimizer, double learningRate,
			boolean fixInputLayer, boolean useOutputBias) {
		HashMap<String, Object> config = new HashMap<String, Object>();
		config.put("class", "MultiLayerPerceptron");
        config.put("lang", lang);
        config.put("embedding", embeddingList);
        config.put("embSizeList", embSizeList);
        config.put("numLayer", numLayer);
        config.put("hiddenSize", hiddenSize);
        config.put("activation", activation);
        config.put("dropout", dropout);
        config.put("optimizer", optimizer);
        config.put("learningRate", learningRate);
        config.put("fixInputLayer", fixInputLayer);
		return config;
	}
	
	/**
	 * Convenience method to generate a config Map from file
	 * @param filename
	 * @return
	 * @throws FileNotFoundException
	 */
	public static HashMap<String, Object> createConfigFromFile(String filename) throws FileNotFoundException {
		Scanner scan = new Scanner(new File(filename));
		HashMap<String, Object> config = new HashMap<String, Object>();
		config.put("class", "MultiLayerPerceptron");
		while(scan.hasNextLine()){
			String line = scan.nextLine().trim();
			if(line.equals("")){
				continue;
			}
			String[] info = line.split(" ");
			if(info[0].equals("serverAddress")) {
			} else if(info[0].equals("serverPort")) {
			} else if(info[0].equals("lang")) {
				config.put("lang", info[1]);
			} else if(info[0].equals("wordEmbedding")) {  //senna glove polygot
				List<String> embeddingList = new ArrayList<String>();
				for (int i = 1; i < info.length; i++) {
					embeddingList.add(info[i]);
				}
				config.put("embedding", embeddingList);
			} else if(info[0].equals("embeddingSize")) {
				List<Integer> embSizeList = new ArrayList<Integer>();
				for (int i = 1; i < info.length; i++) {
					embSizeList.add(Integer.parseInt(info[i])); 
				}
				config.put("embSizeList", embSizeList);
			} else if(info[0].equals("numLayer")) {
				config.put("numLayer", Integer.parseInt(info[1]));
			} else if(info[0].equals("hiddenSize")) {
				config.put("hiddenSize", Integer.parseInt(info[1]));
			} else if(info[0].equals("activation")) { //tanh, relu, identity, hardtanh
				config.put("activation", info[1]);
			} else if(info[0].equals("dropout")) {
				config.put("dropout", Double.parseDouble(info[1]));
			} else if(info[0].equals("optimizer")) {  //adagrad, adam, sgd , none(be careful with the config in statnlp)
				config.put("optimizer", info[1]);
			} else if(info[0].equals("learningRate")) {
				config.put("learningRate", Double.parseDouble(info[1]));
			} else if(info[0].equals("fixInputLayer")) {
				config.put("fixInputLayer", Boolean.parseBoolean(info[1]));
			} else if(info[0].equals("useOutputBias")) {
			} else {
				System.err.println("Unrecognized option: " + line);
			}
		}
		scan.close();
		return config;
	}
	
	/**
	 * Setters and Getters
	 */
	
	public List<String> getEmbeddingList() {
		return embeddingList;
	}
	
	public int getVocabSize() {
		return vocabSize;
	}
	
	public int getHiddenSize() {
		if (numLayer > 0) {
			return hiddenSize;
		} else {
			return totalInputDim;
		}
	}
	
	public int getNumLayer() {
		return numLayer;
	}

	public void resetCount() {
		countOutputMatrix.set(0.0);
		countWeightMatrix.set(0.0);
	}
	
	private double[] getArray(DoubleTensor t) {
		return t.storage().getRawData().getDoubleArray(0, (int) t.nElement());
	}
}
