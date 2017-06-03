package com.statnlp.neural;

import java.util.HashMap;

import com.statnlp.hybridnetworks.FeatureValueProvider;

public abstract class NeuralNetworkFeatureValueProvider extends FeatureValueProvider {
	
	protected HashMap<String,Object> config;
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		this(null, numLabels);
	}
	
	public NeuralNetworkFeatureValueProvider(HashMap<String,Object> config, int numLabels) {
		super(numLabels);
		this.config = config;
	}
	
	@Override
	public void computeValues() {
		forward(isTraining);
	}
	
	@Override
	public void update() {
		backward();
	}
	
	public abstract void forward(boolean training);
	
	public abstract void backward();
	
	public abstract void save(String prefix);
	
	public abstract void load(String prefix);
	
	public abstract void cleanUp();
}
