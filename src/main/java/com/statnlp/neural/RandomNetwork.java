package com.statnlp.neural;

import java.util.Random;

import com.statnlp.hybridnetworks.ContinuousFeature;

public class RandomNetwork extends AbstractNetwork {
	
	private Random rand;
	
	private int randomSize;

	public RandomNetwork(String name, int H) {
		super(name);
		this.randomSize = H;
		this.cfStorage.setDim(H);
	}
	
	public int getRandomSize() {
		return randomSize;
	}

	@Override
	public int getParamSize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double[] getParams() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] getGradParams() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] initialize() {
		// TODO Auto-generated method stub
		rand = new Random(1);
		return getParams();
	}

	@Override
	public void forward(boolean training) {
		for (ContinuousFeature f : cfSet) {
			for (int i = 0; i < randomSize; i++) {
				f.setFv(i, rand.nextDouble());
			}
		}
	}

	@Override
	public void backward() {
		for (ContinuousFeature f : cfSet) {
			for (int i = 0; i < randomSize; i++) {
			}
		}
	}

	@Override
	public void save(String prefix) {
		// TODO Auto-generated method stub

	}

	@Override
	public void load(String prefix) {
		// TODO Auto-generated method stub

	}

	@Override
	public void cleanUp() {
		// TODO Auto-generated method stub

	}

}
