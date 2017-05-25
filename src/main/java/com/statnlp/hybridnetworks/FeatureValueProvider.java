package com.statnlp.hybridnetworks;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public abstract class FeatureValueProvider {
	
	protected double[] weights, gradWeights, params, gradParams;
	
	protected double[] gradOutput;
	
	protected Map<Integer,Map<Integer,Map<Integer,Object>>> edge2input;
	
	protected Set<Object> inputSet;
	
	protected Map<Object,Double> input2score;
	
	public FeatureValueProvider() {
		edge2input = new HashMap<Integer, Map<Integer,Map<Integer,Object>>>();
		input2score = new HashMap<Object, Double>();
		inputSet = new LinkedHashSet<Object>();
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
		inputSet.add(input);
	}
	
	public abstract void initialize();
	
	public abstract void computeValues();
	
	public abstract void update();
	
	public double[] getWeights() {
		return weights;
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
	
	public double getScore(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		Map<Integer, Map<Integer, Object>> tmp = edge2input.get(instanceID);
		if (tmp == null)
			return 0.0;
		
		Map<Integer, Object> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return 0.0;
		
		Object input = tmp2.get(children_k_index);
		if (input == null)
			return 0.0;
		return input2score.get(input);
	}
	
// TODO for backpropagation
//	for(each i)
//		gradWeights[i] += count * output[i];
//		gradOutput[i] += count * weights[i];
//
	
}
