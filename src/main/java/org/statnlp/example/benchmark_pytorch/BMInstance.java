package org.statnlp.example.benchmark_pytorch;

import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseInstance;

public class BMInstance extends BaseInstance<BMInstance, Sentence, List<String>> {

	public BMInstance(int instanceId, double weight) {
		this(instanceId, weight, null, null);
	}
	
	public BMInstance(int instanceId, double weight, Sentence sent, List<String> output) {
		super(instanceId, weight);
		this.input = sent;
		this.output = output;
	}

	private static final long serialVersionUID = 1L;

	@Override
	public int size() {
		return this.input.length();
	}

	public Sentence duplicateInput(){
		return this.input;
	}
	
}
