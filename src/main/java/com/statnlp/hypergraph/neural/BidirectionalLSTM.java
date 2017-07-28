package com.statnlp.hypergraph.neural;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BidirectionalLSTM extends NeuralNetworkCore {
	
	/**
	 * Number of unique input sentences
	 */
	private int numSent;
	
	public BidirectionalLSTM(int hiddenSize, boolean bidirection, String optimizer, int numLabels, int gpuId, String embedding) {
		super(numLabels);
		config.put("class", "SimpleBiLSTM");
        config.put("hiddenSize", hiddenSize);
        config.put("bidirection", bidirection);
        config.put("optimizer", optimizer);
        config.put("numLabels", numLabels);
        config.put("embedding", embedding);
        config.put("gpuid", gpuId);
	}

	@Override
	public int getNNInputSize() {
		Set<String> sentenceSet = new HashSet<String>();
		int maxSentLen = 0;
		for (Object obj : nnInput2Id.keySet()) {
			String sent = (String)obj;
			sentenceSet.add(sent);
			int sentLen = sent.split(" ").length;
			if (sentLen > maxSentLen) {
				maxSentLen = sentLen;
			}
		}
		List<String> sentences = new ArrayList<String>(sentenceSet);
		config.put("sentences", sentences);
		this.numSent = sentences.size();
		System.out.println("maxLen="+maxSentLen);
		System.out.println("#sent="+numSent);
		return numSent*maxSentLen;
	}

	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		return sentAndPos.getKey();
	}
	
	@Override
	public int hyperEdgeInput2OutputRowIndex (Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		int sentID = nnInput2Id.get(sentAndPos.getKey());
		int row = sentAndPos.getValue()*this.numSent+sentID;
		return row;
	}

}
