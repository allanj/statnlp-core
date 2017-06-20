package com.statnlp.example.linear_ne;

import com.statnlp.neural.ContinuousFeatureValueProvider;

public class ECRFContinuousFeatureValueProvider extends ContinuousFeatureValueProvider {

	public ECRFContinuousFeatureValueProvider(int numLabels) {
		super(numLabels);
	}

	@Override
	public double getFeatureValue(Object input) {
		String inputStr = (String)input;
		return inputStr.length();
	}

}
