package com.statnlp.commons.ml.opt;

import java.util.Random;

import com.statnlp.commons.ml.opt.LBFGS.ExceptionWithIflag;
import com.statnlp.commons.types.Instance;

public class GradientDescentMinimizer implements Optimizer {

	public enum LearningAdjustment {
		OPTIMAL, CONSTANT, INVT
	};
	
	private LearningAdjustment learningAdjustment;
	private double eta0; // initial learning rate
	private double gamma = 0.8; // eta0 multiplier
	private int timestep = 10000; // update eta every n timestep
	private int iterations;
	private int batchSize;
	private static Random random = new Random(1);
	private static final boolean CHECK_GRADIENT = false;
	
	private double _obj;
	private double[] _w;
	private double[] _g;
	
	private Instance[] trainingData;
	
	public GradientDescentMinimizer() {
		// defaults to SGD
		this(LearningAdjustment.CONSTANT, 0.01, 1.0, 50, 1);
	}

	public GradientDescentMinimizer(LearningAdjustment learningAdjustment, double eta0, double alpha, int iterations, int batchSize) {
		setLearningAdjustment(learningAdjustment);
		setEta0(eta0);
		setIterations(iterations);
		setBatchSize(batchSize);
		System.out.println("Training data size = "+trainingData.length);
		System.out.println("Learning rate = "+learningAdjustment);
		if (learningAdjustment != LearningAdjustment.INVT)
			System.out.println("eta0 = "+eta0);
		System.out.println("Iterations = "+iterations);
		System.out.println("Batch size = "+batchSize);
	}

	public GradientDescentMinimizer(double eta0, int iterations) {
		this(LearningAdjustment.CONSTANT, eta0, 1.0, iterations, 1);
	}
	
	public LearningAdjustment getLearningRate() {
		return learningAdjustment;
	}

	public void setLearningAdjustment(LearningAdjustment learningRate) {
		this.learningAdjustment = learningRate;
	}

	public double getEta0() {
		return eta0;
	}

	public void setEta0(double eta0) {
		this.eta0 = eta0;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}
	
	
	@Override
	public void setObjective(double f) {
		this._obj = f;
	}

	@Override
	public void setVariables(double[] x) {
		this._w = x;
	}

	@Override
	public void setGradients(double[] g) {
		this._g = g;
	}

	@Override
	public boolean optimize() throws ExceptionWithIflag {
		int dim = _w.length;
		
		return false;
	}

}
