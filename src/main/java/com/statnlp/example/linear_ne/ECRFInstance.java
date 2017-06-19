package com.statnlp.example.linear_ne;

import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.example.base.BaseInstance;


public class ECRFInstance extends BaseInstance<ECRFInstance, Sentence, ArrayList<String>> {

	private static final long serialVersionUID = 1851514046050983662L;
	
	public ECRFInstance(int instanceId, double weight, Sentence sent) {
		super(instanceId, weight);
		this.input = sent;
	}
	
	@Override
	public int size() {
		return this.input.length();
	}
	
	public Sentence duplicateInput(){
		return input;
	}
	
	@SuppressWarnings("unchecked")
	public ArrayList<String> duplicateOutput() {
		return (ArrayList<String>)this.output.clone();
	}
}
