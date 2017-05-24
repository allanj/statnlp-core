package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

import com.statnlp.neural.AbstractInput;

public class ContinuousFeatureStorage implements Serializable {

	private static final long serialVersionUID = -3046249582596299640L;
	
	private HashMap<AbstractInput,Integer> input2row;
	private int dim;
	private double[] fv;
	private double[] fvGrad;
	
	public ContinuousFeatureStorage() {
		this.input2row = new HashMap<AbstractInput, Integer>();
	}
	
	public synchronized int addInput(AbstractInput input) {
		if (!input2row.containsKey(input)) {
			input2row.put(input, input2row.size());
		}
		return input2row.get(input);
	}
	
	public Set<AbstractInput> getInputs() {
		return input2row.keySet();
	}

	public double getFv(int inputID, int idx) {
		return fv[inputID*dim+idx];
	}
	
	public double getFvGrad(int inputID, int idx) {
		return fvGrad[inputID*dim+idx];
	}
	
	public int getDim() {
		return dim;
	}

	public void setDim(int dim) {
		this.dim = dim;
	}

	public void setFv(int inputID, int idx, double val) {
		if (fv == null) {
			fv = new double[input2row.size()*dim];
		}
		fv[inputID*dim+idx] = val;
	}
	
	public void setFvGrad(int inputID, int idx, double val) {
		if (fvGrad == null) {
			fvGrad = new double[input2row.size()*dim];
		}
		fvGrad[inputID*dim+idx] = val;
	}
}
