package com.statnlp.neural;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BidirectionalLSTM extends NeuralNetworkFeatureValueProvider {
	
	/**
	 * Number of unique input sentences
	 */
	private int numSent;
	
	public BidirectionalLSTM(int hiddenSize, boolean bidirection, String optimizer, int numLabels, int gpuId, String embedding) {
		super(numLabels);
		config.put("class", "BidirectionalLSTM");
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
		for (Object obj : fvpInput2id.keySet()) {
			String sent = (String)obj;
			sentenceSet.add(sent);
			int sentLen = sent.split(" ").length;
			if (sentLen > maxSentLen) {
				maxSentLen = sentLen;
			}
		}
		List<String> sentences = new ArrayList<String>(sentenceSet);
		//need to sort the sentences to obtain the same results
		Collections.sort(sentences);
		for (int i = 0; i < sentences.size(); i++) {
			String sent = sentences.get(i);
			fvpInput2id.put(sent, i);
		}
		config.put("sentences", sentences);
		this.numSent = sentences.size();
		System.out.println("maxLen="+maxSentLen);
		System.out.println("#sent="+numSent);
		return numSent*maxSentLen;
	}

	@Override
	public Object edgeInput2FVPInput(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		return sentAndPos.getKey();
	}
	
	@Override
	public int input2Index (Object input) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) input;
		int sentID = fvpInput2id.get(sentAndPos.getKey());
		int row = sentAndPos.getValue()*this.numSent+sentID;
		return row;
	}

}
