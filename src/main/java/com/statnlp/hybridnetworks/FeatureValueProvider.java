package com.statnlp.hybridnetworks;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;


public abstract class FeatureValueProvider {
	
	protected double[] weights, gradWeights;
	
	protected double[] params, gradParams;
	
	protected double[] output, gradOutput;
	
	protected Map<Integer,Map<Integer,Map<Integer,Object>>> edge2input;
	
	protected Map<Object,Integer> input2id;
	
	protected Map<Object,Double> input2score;
	
	public FeatureValueProvider() {
		edge2input = new HashMap<Integer, Map<Integer,Map<Integer,Object>>>();
		input2score = new HashMap<Object, Double>();
		input2id = new LinkedHashMap<Object,Integer>();
	}
	
	public synchronized void addHyperEdgeInput(Network network, int parent_k, int children_k_idx, Object input) {
		int instanceID = network.getInstance().getInstanceId();
		if ( ! edge2input.containsKey(instanceID)) {
			edge2input.put(instanceID, new HashMap<Integer, Map<Integer,Object>>());
		}
		if ( ! edge2input.get(instanceID).containsKey(parent_k)) {
			edge2input.get(instanceID).put(parent_k, new HashMap<Integer,Object>());
		}
		edge2input.get(instanceID).get(parent_k).put(children_k_idx, input);
		input2id.put(input, input2id.size());
	}
	
	public abstract void initialize();
	
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
	
// TODO for backpropagation
//	for(each i)
//		gradWeights[i] += count * output[i];
//		gradOutput[i] += count * weights[i];
//
	
}
