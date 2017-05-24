package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.concurrent.ConcurrentHashMap;

import com.statnlp.neural.AbstractInput;

public class ContinuousFeature implements Serializable {

	private static final long serialVersionUID = -3046249582596299640L;
	
	private ContinuousFeatureIdentifier fb;
	
	private ContinuousFeatureStorage fst;
	
	private int inputID;
	
	public ContinuousFeature(AbstractInput input, int fs, int len, ContinuousFeatureStorage featureStorage) {
		this(new ContinuousFeatureIdentifier(input, fs, len), featureStorage);
	}
	
	public ContinuousFeature(ContinuousFeatureIdentifier featureBox, ContinuousFeatureStorage featureStorage) {
		this.fb = featureBox;
		this.fst = featureStorage;
		this.inputID = this.fst.addInput(this.fb.getInput());
	}
	
	public int getStartFs() {
		return fb.getStartFs();
	}
	
	public void setStartFs(int startFs) {
		fb.setStartFs(startFs);
	}

	public int getLenFs() {
		return fb.getLenFs();
	}
	
	public double getFv(int idx) {
		return fst.getFv(inputID, idx);
	}
	
	public void setFv(int idx, double val) {
		fst.setFv(inputID, idx, val);
	}
	
	public double getFvGrad(int idx) {
		return fst.getFvGrad(inputID, idx);
	}
	
	public void setFvGrad(int idx, double val) {
		fst.setFvGrad(inputID, idx, val);
	}
	
	public AbstractInput getInput() {
		return fb.getInput();
	}
	
	public ContinuousFeatureStorage getStorage() {
		return fst;
	}
	
	@Override
	public int hashCode() {
		return this.fb.hashCode();
	}
	
	@Override
	public boolean equals(Object o){
	    if(o == null)
	    	return false;
	    if(!(o instanceof ContinuousFeature))
	    	return false;

	    ContinuousFeature other = (ContinuousFeature) o;
	    return this.fb.equals(other.fb);
	}
}
