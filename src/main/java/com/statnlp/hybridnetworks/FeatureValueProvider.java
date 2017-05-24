package com.statnlp.hybridnetworks;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;


public abstract class FeatureValueProvider {
	
	protected ContinuousFeatureStorage cfStorage;
	
	protected Set<ContinuousFeature> cfSet;
	
	public FeatureValueProvider(ContinuousFeatureStorage storage) {
		setStorage(storage);
		cfSet = Collections.synchronizedSet(new LinkedHashSet<ContinuousFeature>());
	}
	
	public ContinuousFeatureStorage getStorage() {
		return cfStorage;
	}
	
	public void setStorage(ContinuousFeatureStorage storage) {
		this.cfStorage = storage;
	}
	
	public void addFeature(ContinuousFeature f) {
		cfSet.add(f);
	}
	
	public abstract void initialize();
	
	public abstract void computeValues();
	
	public abstract void update();
	
	public abstract int getParamSize();

	public abstract double[] getParams();

	public abstract double[] getGradParams();
	
}
