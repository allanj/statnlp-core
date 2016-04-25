/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package com.statnlp.commons.ml.opt;


import com.statnlp.commons.ml.opt.LBFGS.ExceptionWithIflag;

/**
 * @author wei_lu
 *
 */
public class GradientDescentOptimizer implements Optimizer{
	
	public static final double DEFAULT_LEARNING_RATE = 0.01;
	
	private double _learningRate;
	private double[] _x;
	private double[] _g;
	private double _obj;
	private double prevOuterProduct[];
	
//	private double _T = 1.0; // Controls learning rate adjustment
	private boolean adaGrad = true;
	
	public GradientDescentOptimizer(int weightLength){
		this(DEFAULT_LEARNING_RATE, weightLength);
	}
	
	public GradientDescentOptimizer(double learningRate,int weightLength){
		this._learningRate = learningRate;
//		this._T = 1;
		this.prevOuterProduct = new double[weightLength];
	}
	
	public double getLearningRate(){
		return this._learningRate;
	}
	
	@Override
	public void setVariables(double[] x){
//		System.err.println("x0="+x[0]);
		this._x = x;
	}
	
	@Override
	public void setObjective(double obj){
		this._obj = obj;
	}
	
	@Override
	public void setGradients(double[] g){
//		System.err.println("g0="+g[0]);
		this._g = g;
	}

	@Override
	public double getObjective() {
		return _obj;
	}

	@Override
	public double[] getVariables() {
		return _x;
	}

	@Override
	public double[] getGradients() {
		return _g;
	}
	
	public boolean optimize() throws ExceptionWithIflag{
//		this._learningRate *= 1/this._T;
		for(int k = 0; k<this._x.length; k++){
			double updateCof = this._learningRate;
			if(adaGrad) {
				prevOuterProduct[k] += this._g[k]*this._g[k];
				if(prevOuterProduct[k]!=0.0)
					updateCof = this._learningRate/Math.sqrt(prevOuterProduct[k]);
			}
			this._x[k] -= updateCof * this._g[k];
		}
//		this._T++;
		return false;
	}
}
