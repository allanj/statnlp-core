package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public class ContinuousFeatureStorage implements Serializable {

	private static final long serialVersionUID = -3046249582596299640L;
	
	private Map<ContinuousFeatureIdentifier,Integer> id2row;
	private int dim;
	private double[] fv;
	private double[] fvGrad;
	
	public ContinuousFeatureStorage() {
		this.id2row = new LinkedHashMap<ContinuousFeatureIdentifier,Integer>();
	}
	
	public synchronized int addInput(ContinuousFeatureIdentifier cfID) {
		if (!id2row.containsKey(cfID)) {
			id2row.put(cfID, id2row.size());
		}
		return id2row.get(cfID);
	}

	public double getFv(int inputID, int idx) {
		return fv[inputID*dim+idx];
	}
	
	public double getFvGrad(int inputID, int idx) {
		return fvGrad[inputID*dim+idx];
	}
	
	public double[] getFvGrad() {
		return fvGrad;
	}
	
	public int getDim() {
		return dim;
	}

	public void setDim(int dim) {
		this.dim = dim;
	}

	public void setFv(int inputID, int idx, double val) {
		lazyInitFv();
		fv[inputID*dim+idx] = val;
	}
	
	public void setFv(double[] vals) {
		lazyInitFv();
		System.arraycopy(vals, 0, fv, 0, id2row.size()*dim);
	}
	
	public void setFv(int inputID, double[] vals) {
		lazyInitFv();
		System.arraycopy(vals, 0, fv, inputID*dim, dim);
	}
	
	public void setFvGrad(int inputID, int idx, double val) {
		lazyInitFvGrad();
		fvGrad[inputID*dim+idx] = val;
	}
	
	public Set<ContinuousFeatureIdentifier> getIdentifiers() {
		return id2row.keySet();
	}
	
	private void lazyInitFv() {
		if(fv == null) {
			fv = new double[id2row.size()*dim];
		}
	}
	
	private void lazyInitFvGrad() {
		if(fvGrad == null) {
			fvGrad = new double[id2row.size()*dim];
		}
	}
}
