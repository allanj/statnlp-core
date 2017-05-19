package com.statnlp.neural;

import java.util.ArrayList;
import java.util.List;

public class NgramInput implements AbstractInput {
	// first dimension: type (word, tag)
	// second dimension: the actual token
	private List<List<String>> ngrams;
	
	public NgramInput(int numType) {
		ngrams = new ArrayList<List<String>>();
		for (int i = 0; i < numType; i++) {
			ngrams.add(new ArrayList<String>());
		}
	}
	
	public void addNgram(int idx, String... tokens) {
		for (String token : tokens) {
			ngrams.get(idx).add(token);
		}
	}
}
