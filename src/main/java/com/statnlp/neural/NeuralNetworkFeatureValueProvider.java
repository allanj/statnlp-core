package com.statnlp.neural;

import java.util.HashMap;

import com.statnlp.hybridnetworks.FeatureValueProvider;

public abstract class NeuralNetworkFeatureValueProvider extends FeatureValueProvider {
	
	/**
	 * The configuration of this neural network
	 */
	protected HashMap<String,Object> config;
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		this(null, numLabels);
	}
	
	public NeuralNetworkFeatureValueProvider(HashMap<String,Object> config, int numLabels) {
		super(numLabels);
		this.config = config;
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
