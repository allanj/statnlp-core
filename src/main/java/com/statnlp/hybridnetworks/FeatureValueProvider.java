package com.statnlp.hybridnetworks;

import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.LinkedHashMap;

import com.statnlp.commons.ml.opt.MathsVector;


public abstract class FeatureValueProvider {
	
	/**
	 * The total number of unique output labels
	 */
	protected int numLabels;
	
	/**
	 * The CRF weights and gradients of the continuous features,
	 * each having the shape (numLabels x embeddingDimension)
	 */
	//protected double[] weights, gradWeights;
	
	/**
	 * The provider's internal weights and gradients
	 */
	protected double[] params, gradParams;
	
	/**
	 * A flattened matrix containing the continuous values
	 * with the shape (inputSize x numLabels).
	 */
	protected double[] output, countOutput;
	
	/**
	 * Maps a hyper-edge tuple (instance ID,parent node ID,child index) to an input-output pair
	 */
	protected TIntObjectHashMap<TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>> edge2io;
	
	/**
	 * Maps an input to a corresponding index
	 */
	protected LinkedHashMap<Object,Integer> input2id;
	
	/**
	 * The coefficient used for regularization, i.e., batchSize/totalInstNum.
	 */
	protected double scale;
	
	/**
	 * Whether we are training or decoding
	 */
	protected boolean isTraining = true;
	
	public FeatureValueProvider(int numLabels) {
		edge2io = new TIntObjectHashMap<TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>>();
		input2id = new LinkedHashMap<Object,Integer>();
		this.numLabels = numLabels;
	}
	
	/**
	 * Add a hyper-edge and its corresponding input-output pair
	 * TODO: fix the synchronization, it will cause the output different everytime.
	 * @param network
	 * @param parent_k
	 * @param children_k_idx
	 * @param input
	 * @param output
	 */
	public synchronized void addHyperEdge(Network network, int parent_k, int children_k_idx, Object input, int output) {
		int instanceID = network.getInstance().getInstanceId();
		if ( ! edge2io.containsKey(instanceID)) {
			edge2io.put(instanceID, new TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>());
		}
		if ( ! edge2io.get(instanceID).containsKey(parent_k)) {
			edge2io.get(instanceID).put(parent_k, new TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>());
		}
		if ( ! edge2io.get(instanceID).get(parent_k).containsKey(children_k_idx)) {
			edge2io.get(instanceID).get(parent_k).put(children_k_idx, new SimpleImmutableEntry<Object,Integer>(input,output));
		}
		if ( ! input2id.containsKey(input)) {
			input2id.put(input, input2id.size());
		}
	}
	
	/**
	 * Initialize this provider (e.g., create a network and prepare its input)
	 */
	public abstract void initialize();
	
	/**
	 * Get the score associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return score
	 */
	public abstract double getScore(Network network, int parent_k, int children_k_index);
	
	/**
	 * Pre-compute all scores for each hyper-edge.
	 * In neural network, this is equivalent to forward.
	 */
	public abstract void initializeScores();
	
	/**
	 * Accumulate count for a specified hyper-edge
	 * @param count
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 */
	public abstract void update(double count, Network network, int parent_k, int children_k_index);
	
	/**
	 * Compute gradient based on the accumulated counts from all hyper-edges.
	 * In neural network, this is equivalent to backward.
	 */
	public abstract void update();
	
	/**
	 * Get the input associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return input
	 */
	public Object getHyperEdgeInput(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>>> tmp = edge2io.get(instanceID);
		if (tmp == null)
			return null;
		
		TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return null;
		
		Object input = tmp2.get(children_k_index).getKey();
		return input;
	}
	
	/**
	 * Get the output label index for a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return output
	 */
	public int getHyperEdgeOutput(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>>> tmp = edge2io.get(instanceID);
		if (tmp == null)
			return -1;
		
		TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return -1;
		
		int output = tmp2.get(children_k_index).getValue();
		return output;
	}
	
	/**
	 * Reset provider input
	 */
	public void clearInput() {
		input2id.clear();
	}
	
	public abstract void closeProvider();
	
	/**
	 * Reset gradient
	 */
	public void resetGrad() {
		if (countOutput != null) {
			Arrays.fill(countOutput, 0.0);
		}
		if (gradParams != null && getParamSize() > 0) {
			Arrays.fill(gradParams, 0.0);
		}
	}
	
	
	public double getL2Params() {
		if (getParamSize() > 0) {
			return MathsVector.square(params);
		}
		return 0.0;
	}
	
	public void addL2ParamsGrad() {
		if (getParamSize() > 0) {
			double _kappa = NetworkConfig.L2_REGULARIZATION_CONSTANT;
			for(int k = 0; k<gradParams.length; k++) {
				if(_kappa > 0) {
					gradParams[k] += 2 * scale * _kappa * params[k];
				}
			}
		}
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
	
	public void setScale(double coef) {
		scale = coef;
	}
	
	public void setTraining(boolean flag) {
		isTraining = flag;
	}
	
	public boolean isTraining() {
		return isTraining;
	}
}
