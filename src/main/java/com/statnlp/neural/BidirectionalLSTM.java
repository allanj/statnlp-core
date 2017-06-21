package com.statnlp.neural;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;

import gnu.trove.map.hash.TObjectIntHashMap;
import scala.util.Random;
import th4j.Tensor.DoubleTensor;

public class BidirectionalLSTM extends NeuralNetworkFeatureValueProvider {
	
	protected HashMap<String,Object> config;
	
	/**
	 * Number of unique input sentences
	 */
	private int numSent;
	
	/**
	 * Maximum sentence length
	 */
	private int maxSentLen;
	
	/**
	 * Map input sentence to index
	 */
	private TObjectIntHashMap<String> sentence2id;

	public BidirectionalLSTM(int numLabels) {
		this(100, true, "none", numLabels);
	}
	
	public BidirectionalLSTM(int hiddenSize, int numLabels) {
		this(hiddenSize, true, "none", numLabels);
	}
	
	public BidirectionalLSTM(int hiddenSize, String optimizer, int numLabels) {
		this(hiddenSize, true, optimizer, numLabels);
	}
	
	public BidirectionalLSTM(int hiddenSize, boolean bidirection, String optimizer, int numLabels) {
		super(numLabels);
		this.optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		this.sentence2id = new TObjectIntHashMap<String>();
		config = new HashMap<>();
		config.put("class", "BidirectionalLSTM");
        config.put("hiddenSize", hiddenSize);
        config.put("bidirection", bidirection);
        config.put("optimizer", optimizer);
        config.put("numLabels", numLabels);
	}

	@Override
	public void initialize() {
		makeInput();
		int inputSize = numSent*maxSentLen;
		if (isTraining) {
			this.countOutput = new double[inputSize * this.numLabels];
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
	        this.countOutputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
		}
		
		// Forward matrices
        this.output = new double[inputSize * this.numLabels];
		
		config.put("isTraining", isTraining);
		
        Object[] args = new Object[3];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.countOutputTensorBuffer;
        Class<?>[] retTypes;
        if (optimizeNeural && isTraining) {
        	retTypes = new Class[]{DoubleTensor.class, DoubleTensor.class};
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
			val = output[row * numLabels + outputLabel];
		}
		return val;
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
			int idx = row * this.numLabels + outputLabel;
			
			synchronized (countOutput) {
				countOutput[idx] -= count;
			}
		}
	}
	
	
}
