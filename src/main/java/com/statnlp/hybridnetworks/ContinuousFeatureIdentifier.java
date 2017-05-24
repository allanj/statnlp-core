package com.statnlp.hybridnetworks;

import com.statnlp.neural.AbstractInput;

public class ContinuousFeatureIdentifier {
	private AbstractInput input;
	private int startFs, lenFs;
	
	public ContinuousFeatureIdentifier(AbstractInput input, int fs, int len) {
		this.input = input;
		this.startFs = fs;
		this.lenFs = len;
	}
	
	public AbstractInput getInput() {
		return input;
	}

	public void setInput(AbstractInput input) {
		this.input = input;
	}
	
	public int getStartFs() {
		return startFs;
	}

	public void setStartFs(int startFs) {
		this.startFs = startFs;
	}

	public int getLenFs() {
		return lenFs;
	}

	public void setLenFs(int lenFs) {
		this.lenFs = lenFs;
	}

	@Override
	public int hashCode() {
		int hash = 1;
		hash = hash * 17 + input.hashCode();
		hash = hash * 31 + startFs;
		hash = hash * 19 + lenFs;
		return hash;
	}
	
	@Override
	public boolean equals(Object o){
	    if(o == null)
	    	return false;
	    if(!(o instanceof ContinuousFeatureIdentifier))
	    	return false;

	    ContinuousFeatureIdentifier other = (ContinuousFeatureIdentifier) o;
	    return this.input.equals(other.input) && this.startFs == other.startFs && this.lenFs == other.lenFs;
	}
	
	@Override
	public String toString(){
		return "[input="+input.toString()+",start="+startFs+",len="+lenFs+"]"; 
	}
}
