package com.statnlp.neural;

import java.util.HashMap;

import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;

import scala.util.Random;
import th4j.Tensor.DoubleTensor;

public abstract class ContinuousFeatureValueProvider extends NeuralNetworkFeatureValueProvider {

	protected HashMap<String,Object> config;
	
	protected int numFeatureValues;
	
	public ContinuousFeatureValueProvider(int numLabels) {
		this(1, numLabels);
	}
	
	public ContinuousFeatureValueProvider(int numFeatureValues ,int numLabels) {
		super(numLabels);
		this.optimizeNeural = true; //for continuous feature, optimize neural is always true.
		this.numFeatureValues = numFeatureValues;
		config = new HashMap<>();
		config.put("class", "ContinuousFeature");
		config.put("numLabels", numLabels);
		config.put("numValues", numFeatureValues);
	}

	@Override
	public void initialize() {
		DoubleTensor inputs = makeInput();
		int inputSize = input2id.size();
		if (isTraining) {
			this.countOutput = new double[inputSize * this.numLabels];
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
	        this.countOutputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
		}
		// Forward matrices
		this.output = new double[inputSize * this.numLabels];
		
		config.put("isTraining", isTraining);
		
        Object[] args = new Object[4];
        args[0] = config;
        args[1] = this.outputTensorBuffer;
        args[2] = this.countOutputTensorBuffer;
        args[3] = inputs;
        Class<?>[] retTypes;
        if (isTraining) {
        	retTypes = new Class[]{DoubleTensor.class, DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "initialize", args, retTypes);
        if(isTraining) {
        	this.paramsTensor = (DoubleTensor) outputs[0];
			this.gradParamsTensor = (DoubleTensor) outputs[1];
			Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			if (this.paramsTensor.nElement() > 0) {
				this.params = getArray(this.paramsTensor, this.params);
				for(int i = 0; i < this.params.length; i++) {
					this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
						NetworkConfig.FEATURE_INIT_WEIGHT;
				}
				this.gradParams = getArray(this.gradParamsTensor, this.gradParams);
			}
		}
	}
	
	/**
	 * Fill the featureValue array using the input object
	 * @param input
	 * @return
	 */
	public abstract void getFeatureValue(Object input, double[] featureValue);
	
	public DoubleTensor makeInput() { 
		double[][] featureValues = new double[input2id.size()][this.numFeatureValues];
		for (Object input : input2id.keySet()) {
			this.getFeatureValue(input, featureValues[input2id.get(input)]);
		}
		DoubleTensor dt = new DoubleTensor(featureValues);
		return dt;
	}

	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int inputId = input2id.get(input);
			val = output[inputId * this.numLabels + outputLabel];
		}
		return val;
	}

	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			int outputLabel = getHyperEdgeOutput(network, parent_k, children_k_index);
			int inputId = input2id.get(input);
			int idx = inputId * this.numLabels + outputLabel;
			synchronized (countOutput) {
				countOutput[idx] -= count;
			}
		}
	}

}
