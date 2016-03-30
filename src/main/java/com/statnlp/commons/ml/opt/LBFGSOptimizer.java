/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

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
package com.statnlp.commons.ml.opt;

import com.statnlp.commons.ml.opt.LBFGS.ExceptionWithIflag;

public class LBFGSOptimizer {

	private int _n;
	private int _m = 4;
	private int[] _iprint = {0,0};
	private int[] _iflag = {0};
	private double _eps = 10e-10;
	private double _xtol = 10e-16;
	private double _f;
	private double[] _diag;
	private double[] _x;
	private double[] _g;
	private boolean _diagco = false;
	
	public void setObjective(double f){
		this._f = f;
	}

	public void setVariables(double[] x){
		this._x = x;
		this._n = x.length;
		if(this._diag == null)
		{
			this._diag = new double[this._n];
			for(int k = 0; k<this._n; k++)
				this._diag[k] = 1.0;
		}
	}
	
	public void setGradients(double[] g){
		this._g = g;
	}
	
	public void updateVariables(double[] x){
		this._x = x;
	}
	
	//return true if it should stop.
	public boolean optimize() throws ExceptionWithIflag{
    	LBFGS.lbfgs(this._n, this._m, this._x, this._f, this._g, this._diagco, this._diag, this._iprint, this._eps, this._xtol, this._iflag);
    	return _iflag[0] == 0;
	}
    
}
