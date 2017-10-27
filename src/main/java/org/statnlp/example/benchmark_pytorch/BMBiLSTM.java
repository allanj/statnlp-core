package org.statnlp.example.benchmark_pytorch;

import java.util.AbstractMap.SimpleImmutableEntry;

import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class BMBiLSTM extends NeuralNetworkCore {

	private static final long serialVersionUID = 2893976240095976474L;

	public BMBiLSTM(int numLabels, int hiddenSize, int embeddingSize) {
		super(numLabels);
		this.config.put("class", "BMLSTM");
		this.config.put("hiddenSize", hiddenSize);
		this.config.put("embeddingSize", embeddingSize);
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
