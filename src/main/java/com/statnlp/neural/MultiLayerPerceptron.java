package com.statnlp.neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
public class MultiLayerPerceptron extends AbstractNetwork {
	private boolean DEBUG = false;
	
	public static final String IN_SEP = "#IN#";
	public static final String OUT_SEP = "#OUT#";
	
	// Torch NN server information
	private LuaState L;
	private DoubleTensor paramsTensor, gradParamsTensor;
	private DoubleTensor outputTensorBuffer, gradTensorBuffer;
	
	private SimpleMatrix weightMatrix, gradWeightMatrix;
	private SimpleMatrix outputMatrix, gradOutputMatrix;
	private SimpleMatrix forwardMatrix;
	private SimpleMatrix countWeightMatrix, countOutputMatrix;
	
	private boolean optimizeNeural;
	private boolean hasParams;
	
	private List<String> embedding;
	private int vocabSize;
	private int hiddenSize, numLayer;
	private int totalInputDim;
	
	private List<Integer> inputDimList;
	private List<String> wordList;
	private List<HashMap<String,Integer>> token2idxList;

	private SimpleMatrix weightMatrix_tran;

	public MultiLayerPerceptron(String name, HashMap<String, Object> config, int numOutput) {
		super(name, config, numOutput);
		this.optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		
		configure();
		this.L = new LuaState();
		this.L.openLibs();
		
		try {
			this.L.load(Files.newInputStream(Paths.get("nn-crf-interface/neural_server/NetworkInterface.lua")),"NetworkInterface.lua");
		} catch (IOException e) {
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
			e.printStackTrace();
		}
		Native.loadLibrary("libjnlua5.1.jnilib", Library.class);
	}
	
	@SuppressWarnings("unchecked")
	public void initialize() {
		Random r = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
		this.embedding = (List<String>) config.get("embedding");
		this.hiddenSize = (int) config.get("hiddenSize");
		this.numLayer = (int) config.get("numLayer");
		
		List<Integer> numInputList = new ArrayList<Integer>();
		this.inputDimList = new ArrayList<Integer>();
		this.wordList = new ArrayList<String>();
		List<List<Integer>> vocab = new ArrayList<List<Integer>>();
		this.token2idxList = new ArrayList<HashMap<String,Integer>>();
		List<String> embeddingList = getEmbeddingList();
		boolean first = true;
		for (Object obj : input2id.keySet()) {
			String input = (String) obj;
			String[] inputPerType = input.split(OUT_SEP);
			List<Integer> entry = new ArrayList<Integer>();
			for (int i = 0; i < inputPerType.length; i++) {
				String[] tokens = inputPerType[i].split(IN_SEP);
				if (first) {
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
		
		totalInputDim = 0;
		List<Integer> embSizeList = (List<Integer>) config.get("embSizeList");
		for (int i = 0; i < token2idxList.size(); i++) {
			if (embSizeList.get(i) != 0) {
				totalInputDim += embSizeList.get(i);
			} else {
				totalInputDim += token2idxList.get(i).size();
			}
		}

		// Weight matrix
		this.weightMatrix = new SimpleMatrix(numOutput, getHiddenSize());
		this.weightMatrix_tran = new SimpleMatrix(getHiddenSize(), numOutput);
		this.gradWeightMatrix = new SimpleMatrix(numOutput, getHiddenSize());
		DMatrixRMaj weightDMatrix = this.weightMatrix.getMatrix();
		this.weights = weightDMatrix.data;
		for(int i = 0; i < weights.length; i++) {
			weights[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (r.nextDouble()-.5)/10 :
				NetworkConfig.FEATURE_INIT_WEIGHT;
		}
		DMatrixRMaj gradWeightDMatrix = this.gradWeightMatrix.getMatrix();
		this.gradWeights = gradWeightDMatrix.data;
		
		config.put("numInputList", numInputList);
        config.put("inputDimList", inputDimList);
        config.put("wordList", wordList);
        config.put("vocab", vocab);
        this.vocabSize = ((List<?>) config.get("vocab")).size();
        
        // 2D buffer array to be used by backward()
        this.outputTensorBuffer = new DoubleTensor(getVocabSize(), getHiddenSize());
        this.gradTensorBuffer = new DoubleTensor(getVocabSize(), getHiddenSize());
        
        // Forward matrices
        this.outputMatrix = new SimpleMatrix(getVocabSize(), getHiddenSize());
        this.forwardMatrix = new SimpleMatrix(getVocabSize(), numOutput);
        
        // Backward matrices
        this.gradOutputMatrix = new SimpleMatrix(getVocabSize(), getHiddenSize());
        DMatrixRMaj gradOutputDMatrix = this.gradOutputMatrix.getMatrix();
        this.gradOutput = gradOutputDMatrix.data;
        this.countOutputMatrix = new SimpleMatrix(getVocabSize(), numOutput);
        this.countWeightMatrix = new SimpleMatrix(numOutput, getVocabSize());
        
        Object[] args = new Object[3];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.gradTensorBuffer;
        Class<?>[] retTypes;
        if (optimizeNeural) {
        	retTypes = new Class[]{DoubleTensor.class,DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "initialize", args, retTypes);
        
		if(optimizeNeural) {
			this.paramsTensor = (DoubleTensor) outputs[0];
			this.gradParamsTensor = (DoubleTensor) outputs[1];
			if (this.paramsTensor.nElement() > 0) {
				this.params = getArray(this.paramsTensor);
				for(int i = 0; i < this.params.length; i++) {
					this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (r.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
				this.gradParams = getArray(this.gradParamsTensor);
				this.hasParams = true;
			}
		}
	}
	
	@Override
	public void initializeForDecoding() {
		setTraining(false);
		List<String> wordList = new ArrayList<String>(this.wordList);
		List<Integer> inputDimList = new ArrayList<Integer>(this.inputDimList);
		List<List<Integer>> vocab = new ArrayList<List<Integer>>();
		List<HashMap<String,Integer>> token2idxList = new ArrayList<HashMap<String,Integer>>();
		for (int i = 0; i < this.token2idxList.size(); i++) {
			token2idxList.add(new HashMap<String, Integer>(this.token2idxList.get(i)));
		}
		List<String> embeddingList = getEmbeddingList();
		for (Object obj : testInput2id.keySet()) {
			String input = (String) obj;
			String[] inputPerType = input.split(OUT_SEP);
			List<Integer> entry = new ArrayList<Integer>();
			for (int i = 0; i < inputPerType.length; i++) {
				String[] tokens = inputPerType[i].split(IN_SEP);
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
		}
		
		config.put("inputDimList", inputDimList);
        config.put("wordList", wordList);
        config.put("vocab", vocab);
        
        // Forward matrices
        this.outputMatrix = new SimpleMatrix(vocab.size(), getHiddenSize());
        this.forwardMatrix = new SimpleMatrix(vocab.size(), numOutput);
        
        Object[] args = new Object[]{config};
        Class<?>[] retTypes = new Class[]{};
        LuaFunctionHelper.execLuaFunction(this.L, "initializeForDecoding", args, retTypes);
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		int instanceID = network.getInstance().getInstanceId();
		boolean isTest = instanceID > 0 && !network.getInstance().isLabeled();
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int H = getHiddenSize();
			Map<Object,Integer> input2id = isTest ? this.testInput2id : this.input2id;
			int id = input2id.get(input);
//			for (int i = 0; i < H; i++) {
//				val += weights[outputLabel*H+i] * output[id*H+i];
//			}
			val = forwardMatrix.get(id, outputLabel);
		}
		return val;
	}
	
	@Override
	public void forward(boolean training) {
		if (optimizeNeural) { // update with new params
			if (hasParams)
				this.paramsTensor.storage().copy(this.params); // we can do this because params is contiguous
		}
		
		Object[] args = new Object[]{training};
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
			int H = getHiddenSize();
			int id = input2id.get(input);
			synchronized (countOutputMatrix) {
				double countOutput = countOutputMatrix.get(id, outputLabel);
				countOutputMatrix.set(id, outputLabel, countOutput+count);
			}
			synchronized (countWeightMatrix) {
				double countWeight = countWeightMatrix.get(outputLabel, id);
				countWeightMatrix.set(outputLabel, id, countWeight-count);
			}
//			for (int i = 0; i < H; i++) {
//				gradOutput[id*H+i] += count * weights[outputLabel*H+i];
//			}
//			for (int i = 0; i < H; i++) {
//				gradWeights[outputLabel*H+i] -= count * output[id*H+i];
//			}
		}
	}
	
	@Override
	public void backward() {
		gradTensorBuffer.storage().copy(gradOutput);
		
		CommonOps_DDRM.mult(countOutputMatrix.getMatrix(), weightMatrix.getMatrix(), gradOutputMatrix.getMatrix());
		CommonOps_DDRM.mult(countWeightMatrix.getMatrix(), outputMatrix.getMatrix(), gradWeightMatrix.getMatrix());
		
		Object[] args = new Object[0];
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(optimizeNeural && hasParams) { // copy gradParams computed by Torch
			gradParams = getArray(this.gradParamsTensor);
		}
		
//		Arrays.fill(gradOutput, 0.0);
		gradOutputMatrix.set(0.0);
		countOutputMatrix.set(0.0);
		countWeightMatrix.set(0.0);
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
	
	/*
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
		nn.forwardNetwork(true);
		nn.backwardNetwork();
	}
	*/
	
	public static HashMap<String, Object> createConfig(
			String lang, List<String> embeddingList, List<Integer> embSizeList,
			int numLayer, int hiddenSize, String activation,
			double dropout, String optimizer, double learningRate,
			boolean fixEmbedding, boolean useOutputBias) {
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
        config.put("fixEmbedding", fixEmbedding);
		return config;
	}
	
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
			} else if(info[0].equals("fixEmbedding")) {
				config.put("fixEmbedding", Boolean.parseBoolean(info[1]));
			} else if(info[0].equals("useOutputBias")) {
			} else {
				System.err.println("Unrecognized option: " + line);
			}
		}
		scan.close();
		return config;
	}
	
	public List<String> getEmbeddingList() {
		return embedding;
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

	private double[] getArray(DoubleTensor t) {
		return t.storage().getRawData().getDoubleArray(0, (int) t.nElement());
	}
}
