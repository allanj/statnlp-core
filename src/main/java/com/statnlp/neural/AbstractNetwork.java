package com.statnlp.neural;

import java.util.HashMap;

import com.statnlp.hybridnetworks.FeatureValueProvider;

public abstract class AbstractNetwork extends FeatureValueProvider {
	
	protected HashMap<String,Object> config;
	
	protected String name;

	protected static int numNetworks;
	
	protected boolean isTraining = true;
	
	public AbstractNetwork(int numOutput) {
		this(""+numNetworks, null, numOutput);
		numNetworks++;
	}
	
	public AbstractNetwork(String name, int numOutput) {
		this(name, null, numOutput);
	}
	
	public AbstractNetwork(HashMap<String,Object> config, int numOutput) {
		this(""+numNetworks, config, numOutput);
		numNetworks++;
	}
	
	public AbstractNetwork(String name, HashMap<String,Object> config, int numOutput) {
		super(numOutput);
		this.name = name;
		this.config = config;
	}
	
	@Override
	public void computeValues() {
		forward(isTraining);
	}
	
	@Override
	public void update() {
		backward();
	}
	
	public abstract void forward(boolean training);
	
	public abstract void backward();
	
	public abstract void save(String prefix);
	
	public abstract void load(String prefix);
	
	public abstract void cleanUp();
	
	public String getName() {
		return this.name;
	}
	
	public void setTraining(boolean flag) {
		isTraining = flag;
	}
	
	public boolean isTraining() {
		return isTraining;
	}
}
