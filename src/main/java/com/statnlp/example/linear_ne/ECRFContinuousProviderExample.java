package com.statnlp.example.linear_ne;

import com.statnlp.hybridnetworks.FeatureValueProvider;
import com.statnlp.hybridnetworks.Network;

public class ECRFContinuousProviderExample extends FeatureValueProvider {

	public ECRFContinuousProviderExample(int numLabels) {
		super(numLabels);
	}

	@Override
	public void initialize() {
		weights = new double[1];
		gradWeights = new double[1];
	}

	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
//			System.out.println("input="+input+",value="+input2score.get(input));
			return input2score.get(input);
		}
		return 0;
	}

	@Override
	public void initializeScores() {
		for (Object input : input2id.keySet()) {
			input2score.put(input, ((String) input).length());
		}
	}

	@Override
	public void update(double count, Network network, int parent_k,
			int children_k_index) {
		Object input = getHyperEdgeInput(network, parent_k, children_k_index);
		if (input != null) {
			synchronized (gradWeights) {
				gradWeights[0] -= count * input2score.get(input);	
			}
		}
	}

	@Override
	public void update() {
	}

}
