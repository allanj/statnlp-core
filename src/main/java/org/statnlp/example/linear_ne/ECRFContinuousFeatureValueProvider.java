package org.statnlp.example.linear_ne;

import org.statnlp.hypergraph.neural.ContinuousFeatureValueProvider;

public class ECRFContinuousFeatureValueProvider extends ContinuousFeatureValueProvider {

	public ECRFContinuousFeatureValueProvider(int numFeatureValues, int numLabels) {
		super(numFeatureValues, numLabels);
	}
	
	public ECRFContinuousFeatureValueProvider(int numLabels) {
		super(numLabels);
	}

	@Override
	public void getFeatureValue(Object input, double[] featureValue) {
		String inputStr = (String)input;
		double val2 = 0.2;
		if (inputStr.length() > 5){
			val2 = 0.8;
		}
		featureValue[0] = inputStr.length();
		featureValue[1] = val2;
	}

}
