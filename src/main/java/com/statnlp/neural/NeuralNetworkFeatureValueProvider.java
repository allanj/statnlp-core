package com.statnlp.neural;

import com.statnlp.hybridnetworks.FeatureValueProvider;

public abstract class NeuralNetworkFeatureValueProvider extends FeatureValueProvider {
	
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		super(numLabels);
	}
	
	
	@Override
	public void initializeScores() {
		forward();
	}
	
	@Override
	public void update() {
		backward();
	}
	
	/**
	 * Neural network's forward
	 */
	public abstract void forward();
	
	/**
	 * Neural network's backpropagation
	 */
	public abstract void backward();
	
	/**
	 * Save the trained model
	 * @param prefix
	 */
	public abstract void save(String prefix);
	
	/**
	 * Load a model from disk
	 * @param prefix
	 */
	public abstract void load(String prefix);
	
	/**
	 * Clean up resources
	 */
	public abstract void cleanUp();
}
