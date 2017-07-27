package com.statnlp.hypergraph.neural;

public abstract class ContinuousFeatureValueProvider extends NeuralNetworkCore {

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
		int inputSize = nnInput2Id.size();
		return inputSize;
	}
	
	/**
	 * Fill the featureValue array using the input object
	 * @param input
	 * @return
	 */
	public abstract void getFeatureValue(Object input, double[] featureValue);
	
	public double[][] makeInput() { 
		double[][] featureValues = new double[nnInput2Id.size()][this.numFeatureValues];
		for (Object input : nnInput2Id.keySet()) {
			this.getFeatureValue(input, featureValues[nnInput2Id.get(input)]);
		}
		return featureValues;
	}
	
	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		return edgeInput;
	}

	@Override
	public int edgeInput2Index(Object input) {
		return nnInput2Id.get(input);
	}

}
