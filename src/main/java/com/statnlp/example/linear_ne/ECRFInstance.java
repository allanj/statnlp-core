package com.statnlp.example.linear_ne;

import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.example.base.BaseInstance;


public class ECRFInstance extends BaseInstance<ECRFInstance, Sentence, ArrayList<String>> {


	private static final long serialVersionUID = 1851514046050983662L;
	protected Sentence sentence;
	protected ArrayList<String> entities;
	protected ArrayList<String> predictons;
	
	protected double predictionScore;
	
	public ECRFInstance(int instanceId, double weight, Sentence sent) {
		super(instanceId, weight);
		this.sentence = sent;
	}
	
	@Override
	public int size() {
		return this.sentence.length();
	}

	@SuppressWarnings({"unchecked"})
	@Override
	public ECRFInstance duplicate() {
		ECRFInstance inst = new ECRFInstance(this._instanceId, this._weight,this.sentence);
		if(entities!=null){
			inst.entities = (ArrayList<String>)entities.clone();
		} else {
			inst.entities = null;
		}
		if(predictons!=null){
			inst.predictons =(ArrayList<String>)predictons.clone();
		} else {
			inst.predictons = null;
		}
		return inst;
	}

	@Override
	public void removeOutput() {
	}

	@Override
	public void removePrediction() {
	}

	@Override
	public Sentence getInput() {
		return this.sentence;
	}

	@Override
	public ArrayList<String> getOutput() {
		return this.entities;
	}

	@Override
	public ArrayList<String> getPrediction() {
		return this.predictons;
	}

	@Override
	public boolean hasOutput() {
		if(entities!=null) return true;
		return false;
	}

	@Override
	public boolean hasPrediction() {
		return false;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void setPrediction(Object o) {
		this.predictons = (ArrayList<String>)o;
	}
	
	public void setPredictionScore(double score){
		this.predictionScore = score;
	}
	
	public double getPredictionScore(){
		return this.predictionScore;
	}
	
}
