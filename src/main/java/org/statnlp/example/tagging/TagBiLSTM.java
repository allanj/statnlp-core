package org.statnlp.example.tagging;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class TagBiLSTM extends NeuralNetworkCore {

	public TagBiLSTM(int numLabels) {
		super(numLabels);
		this.config.put("class", "TagBiLSTM");
		this.config.put("hiddenSize", 100);
	}

	@SuppressWarnings("unchecked")
	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		SimpleImmutableEntry<String, Integer> eInput = (SimpleImmutableEntry<String, Integer>)edgeInput;
		return eInput.getKey();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public int hyperEdgeInput2OutputRowIndex(Object edgeInput) {
		SimpleImmutableEntry<String, Integer> eInput = (SimpleImmutableEntry<String, Integer>)edgeInput;
		int position = eInput.getValue();
		return position * this.getNNInputSize() + this.getNNInputID(eInput.getKey());
	}

}
