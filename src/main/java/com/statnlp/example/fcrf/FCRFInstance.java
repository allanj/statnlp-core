package com.statnlp.example.fcrf;

import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.example.base.BaseInstance;


public class FCRFInstance extends BaseInstance<FCRFInstance, Sentence, ArrayList<String>> {

	private static final long serialVersionUID = 1851514046050983662L;
	protected Sentence sentence;
	protected ArrayList<String> chunks;
	protected ArrayList<String> predictons;
	protected ArrayList<String> chunkPredictons;
	protected ArrayList<String> tagPredictons;
	
	public FCRFInstance(int instanceId, double weight, Sentence sent) {
		super(instanceId, weight);
		this.sentence = sent;
	}

	@Override
	public int size() {
		return this.sentence.length();
	}

	@SuppressWarnings("unchecked")
	@Override
	public FCRFInstance duplicate() {
		FCRFInstance inst = new FCRFInstance(this._instanceId, this._weight,this.sentence);
		if(chunks!=null)
			inst.chunks = (ArrayList<String>)chunks.clone();
		else inst.chunks = null;
		if(predictons!=null)
			inst.predictons =(ArrayList<String>)predictons.clone();
		else inst.predictons = null;
		if(chunkPredictons!=null)
			inst.chunkPredictons =(ArrayList<String>)chunkPredictons.clone();
		else inst.chunkPredictons = null;
		if(tagPredictons!=null)
			inst.tagPredictons =(ArrayList<String>)tagPredictons.clone();
		else inst.tagPredictons = null;
		return inst;
	}

	@Override
	public void removeOutput() {
	}

	@Override
	public void removePrediction() {
	}

	@Override
	public boolean hasPrediction() {
		return false;
	}

	public ArrayList<String> getChunkPredictons() {
		return chunkPredictons;
	}

	public void setChunkPredictons(ArrayList<String> entityPredictons) {
		this.chunkPredictons = entityPredictons;
	}

	public ArrayList<String> getTagPredictons() {
		return tagPredictons;
	}

	public void setTagPredictons(ArrayList<String> tagPredictons) {
		this.tagPredictons = tagPredictons;
	}
	
	public void setChunks(ArrayList<String> entities) {
		this.chunks = entities;
	}
	

}
