package com.statnlp.neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;

import th4j.Tensor.DoubleTensor;

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
	 * A special identifier for unknown input
	 */
	private static final Integer UNKNOWN = -1;
	
	
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
	
	protected HashMap<String, Object> config;
	
	public MultiLayerPerceptron(String configFile, int numLabels) {
		super(numLabels);
		this.config = this.createConfigFromFile(configFile);
		config.put("numLabels", numLabels);
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
		
		if (isTraining) {
			totalInputDim = 0;
			List<Integer> embSizeList = (List<Integer>) config.get("embSizeList");
			for (int i = 0; i < token2idxList.size(); i++) {
				if (embSizeList.get(i) != 0) {
					totalInputDim += embSizeList.get(i) * numInputList.get(i);
				} else {
					totalInputDim += token2idxList.get(i).size() * numInputList.get(i);
				}
			}
			
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(getVocabSize(), this.numLabels);
	        this.countOutputTensorBuffer = new DoubleTensor(getVocabSize(), this.numLabels);
	        // Backward matrices
	        this.countOutput = new double[getVocabSize() * this.numLabels];
		}
		
		this.output = new double[getVocabSize() * this.numLabels];
        
        config.put("isTraining", isTraining);
        
        Object[] args = new Object[3];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.countOutputTensorBuffer;
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
			Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			if (this.paramsTensor.nElement() > 0) {
				this.params = this.getArray(this.paramsTensor, this.params);
				for(int i = 0; i < this.params.length; i++) {
					this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
				this.gradParams = this.getArray(this.gradParamsTensor, this.gradParams);
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
			boolean isUnknown = false;
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
						if (isTraining) {
							inputDimList.set(i, inputDimList.get(i)+1);
							int idx = NetworkConfig.IS_INDEXED_NEURAL_FEATURES? Integer.parseInt(token):token2idx.size();
							token2idx.put(token, idx);
							if (embeddingList.get(i).equals("glove")
								|| embeddingList.get(i).equals("polyglot")) {
								wordList.add(token);
							}
						} else {
							// Unseen
							input2id.put(input, UNKNOWN);
							boolean startDecrement = false;
							for (Object _obj : input2id.keySet()) {
								String _input = (String) _obj;
								if (startDecrement) {
									input2id.put(_input, input2id.get(_input)-1);
								}
								if (_input.equals(input)) {
									startDecrement = true;
								}
							}
							isUnknown = true;
							break;
						}
					}
					int idx = token2idx.get(token);
					if (NetworkConfig.NEURAL_BACKEND.startsWith("torch")) { // 1-indexing
						idx++;
					}
					entry.add(idx);
				}
				if (isUnknown) {
					break;
				}
			}
			if (!isUnknown) {
				vocab.add(entry);
			}
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
			if (id != UNKNOWN) {
				val = output[id * numLabels + outputLabel];
			}
		}
		return val;
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int id = input2id.get(input);
			int idx = id * this.numLabels + outputLabel;
			synchronized (countOutput) {
				countOutput[idx] -= count;
			}
		}
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
	private HashMap<String, Object> createConfigFromFile(String filename) {
		Scanner scan;
		try {
			scan = new Scanner(new File(filename));
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
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return null;
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

	@Override
	public int input2Index(Object input) {
		// TODO Auto-generated method stub
		return 0;
	}

}
