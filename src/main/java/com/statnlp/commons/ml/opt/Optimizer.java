package com.statnlp.commons.ml.opt;

import com.statnlp.commons.ml.opt.LBFGS.ExceptionWithIflag;

public interface Optimizer {

	public void setObjective(double f);

	public void setVariables(double[] x);
	
	public void setGradients(double[] g);
	
	public boolean optimize() throws ExceptionWithIflag;
}