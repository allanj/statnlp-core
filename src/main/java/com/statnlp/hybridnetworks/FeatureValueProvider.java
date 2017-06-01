package com.statnlp.hybridnetworks;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;


public abstract class FeatureValueProvider {
	
	protected int outputIdx, numOutput;
	
	protected double[] weights, gradWeights;
	
	protected double[] params, gradParams;
	
	protected double[] output, gradOutput;
	
	protected Map<Integer,Map<Integer,Map<Integer,Object>>> edge2input;
	
	protected Map<Integer,Map<Integer,Map<Integer,Integer>>> edge2output;
	
	protected Map<Object,Integer> input2id, testInput2id;
	
	protected Map<Object,Double> input2score;
	
	public FeatureValueProvider(int numOutput) {
		edge2input = new HashMap<Integer, Map<Integer,Map<Integer,Object>>>();
		edge2output = new HashMap<Integer, Map<Integer,Map<Integer,Integer>>>();
		input2score = new HashMap<Object, Double>();
		input2id = new LinkedHashMap<Object,Integer>();
		testInput2id = new LinkedHashMap<Object, Integer>();
		this.numOutput = numOutput;
	}
	
	public synchronized void addHyperEdge(Network network, int parent_k, int children_k_idx, Object input, int output) {
		int instanceID = network.getInstance().getInstanceId();
		boolean isTest = instanceID > 0 && !network.getInstance().isLabeled();
		if ( ! edge2input.containsKey(instanceID)) {
			edge2input.put(instanceID, new HashMap<Integer, Map<Integer,Object>>());
			edge2output.put(instanceID, new HashMap<Integer, Map<Integer,Integer>>());
		}
		if ( ! edge2input.get(instanceID).containsKey(parent_k)) {
			edge2input.get(instanceID).put(parent_k, new HashMap<Integer,Object>());
			edge2output.get(instanceID).put(parent_k, new HashMap<Integer,Integer>());
		}
		if ( ! edge2input.get(instanceID).get(parent_k).containsKey(children_k_idx)) {
			edge2input.get(instanceID).get(parent_k).put(children_k_idx, input);
			edge2output.get(instanceID).get(parent_k).put(children_k_idx, output);
		}
		if (!isTest) {
			if ( ! input2id.containsKey(input)) {
				input2id.put(input, input2id.size());
			}
		} else {
			if ( ! testInput2id.containsKey(input)) {
				testInput2id.put(input, testInput2id.size());
			}
		}
	}
	
	public abstract void initialize();
	
	public abstract void initializeForDecoding();
	
	public abstract void computeValues();
	
	public abstract void update(double count, Network network, int parent_k, int children_k_index);
	
	public abstract void update();
	
	public int getWeightSize() {
		return weights.length;
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public double[] getGradWeights() {
		return gradWeights;
	}
	
	public int getParamSize() {
		return params == null ? 0 : params.length;
	}

	public double[] getParams() {
		return params;
	}

	public double[] getGradParams() {
		return gradParams;
	}
	
	public abstract double getScore(Network network, int parent_k, int children_k_index);
	
	public Object getHyperEdgeInput(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		Map<Integer, Map<Integer, Object>> tmp = edge2input.get(instanceID);
		if (tmp == null)
			return null;
		
		Map<Integer, Object> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return null;
		
		Object input = tmp2.get(children_k_index);
		return input;
	}
	
	public int getHyperEdgeOutput(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		Map<Integer, Map<Integer, Integer>> tmp = edge2output.get(instanceID);
		if (tmp == null)
			return -1;
		
		Map<Integer, Integer> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return -1;
		
		int output = tmp2.get(children_k_index);
		return output;
	}
	
	public void clearDecodingState() {
		testInput2id.clear();
	}
}
