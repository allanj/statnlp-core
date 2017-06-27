package com.statnlp.neural;

public abstract class ContinuousFeatureValueProvider extends NeuralNetworkFeatureValueProvider {

	protected int numFeatureValues;
	
	public ContinuousFeatureValueProvider(int numLabels) {
		this(1, numLabels);
	}
	
	public ContinuousFeatureValueProvider(int numFeatureValues ,int numLabels) {
		super(numLabels);
		this.optimizeNeural = true; //for continuous feature, optimize neural is always true.
		this.numFeatureValues = numFeatureValues;
		config.put("class", "ContinuousFeature");
		config.put("numLabels", numLabels);
		config.put("numValues", numFeatureValues);
	}

	@Override
	public int getNNInputSize() {
		double[][] inputs = makeInput();
		config.put("inputs", inputs);
		int inputSize = fvpInput2id.size();
		return inputSize;
	}
	
	/**
	 * Fill the featureValue array using the input object
	 * @param input
	 * @return
	 */
	public abstract void getFeatureValue(Object input, double[] featureValue);
	
	public double[][] makeInput() { 
		double[][] featureValues = new double[fvpInput2id.size()][this.numFeatureValues];
		for (Object input : fvpInput2id.keySet()) {
			this.getFeatureValue(input, featureValues[fvpInput2id.get(input)]);
		}
		return featureValues;
	}
	
	@Override
	public Object edgeInput2FVPInput(Object edgeInput) {
		return edgeInput;
	}

	@Override
	public int input2Index(Object input) {
		return fvpInput2id.get(input);
	}

}
