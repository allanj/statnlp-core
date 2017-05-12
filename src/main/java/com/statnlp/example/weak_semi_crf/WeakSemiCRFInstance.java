package com.statnlp.example.weak_semi_crf;

import java.util.List;

import com.statnlp.example.base.BaseInstance;

public class WeakSemiCRFInstance extends BaseInstance<WeakSemiCRFInstance, String, List<Span>> {
	
	private static final long serialVersionUID = -5338701879189642344L;
	
	public WeakSemiCRFInstance(int instanceId, String input, List<Span> output){
		this(instanceId, 1.0, input, output);
	}
	
	public WeakSemiCRFInstance(int instanceId, double weight) {
		this(instanceId, weight, null, null);
	}
	
	public WeakSemiCRFInstance(int instanceId, double weight, String input, List<Span> output){
		super(instanceId, weight);
		this.input = input;
		this.output = output;
	}
	
	private String[] inputArray = null;
	
	public String[] getInputAsArray(){
		if(inputArray == null){
			inputArray = new String[input.length()];
			for(int i=0; i<input.length(); i++){
				inputArray[i] = input.substring(i, i+1);
			}
		}
		return inputArray;
	}

	@Override
	public int size() {
		return getInput().length();
	}

	public String toString(){
		StringBuilder builder = new StringBuilder();
		builder.append(getInstanceId()+":");
		builder.append(input);
		if(hasOutput()){
			builder.append("\n");
			for(Span span: output){
				builder.append(span+"|");
			}
		}
		return builder.toString();
	}
}
