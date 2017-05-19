package com.statnlp.neural;

import java.util.List;

import com.statnlp.commons.types.Instance;

public abstract class AbstractNetwork {
	// Reference to controller instance for updating weights and getting gradients
	protected NNCRFInterface controller;
	
	// whether to use CRF's optimizer to optimize internal neural parameters
	protected boolean optimizeNeural;
	
	public AbstractNetwork() {
		this(false);
	}
	
	public AbstractNetwork(boolean optimizeNeural) {
		this.optimizeNeural = optimizeNeural;
	}
	
	public void setController(NNCRFInterface controller) {
		this.controller = controller;
	}
	
	public abstract double[] initNetwork(List<Integer> numInputList, List<Integer> inputDimList, List<String> wordList,
			   String lang, List<String> embeddingList, List<Integer> embSizeList,
			   List<Integer> outputDimList, List<List<Integer>> vocab);
	
	public abstract void forwardNetwork(boolean training);
	
	public abstract void backwardNetwork();
	
	public abstract void saveNetwork(String prefix);
	
	public abstract void loadNetwork(String prefix);
	
	public abstract void cleanUp();

	public abstract void setInput(Instance[] instances);

	public abstract int getParamSize();

	public abstract double[] getParams();

	public abstract double[] getGradParams();
}
