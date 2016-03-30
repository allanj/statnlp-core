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
package com.statnlp.commons.types;

import java.io.Serializable;

/**
 * A base class representing an instance, to hold the surface form (e.g., the words of a sentence) of a 
 * training or test instance.<br>
 * This instance can be converted into a {@link Network} using the {@link NetworkCompiler}.<br>
 * Note that it is important to call the {@link #setLabeled()} method on training data, as otherwise this 
 * instance will not be considered during training
 * @author Wei Lu <luwei@statnlp.com>
 *
 */
public abstract class Instance implements Serializable{
	
	private static final long serialVersionUID = 4998596827132890817L;
	
	protected int _instanceId;
	protected double _weight;
	protected boolean _isLabeled;
	
	/**
	 * Create an instance.
	 * The instance id should not be zero.
	 * @param instanceId
	 * @param weight
	 */
	public Instance(int instanceId, double weight){
		if(instanceId==0)
			throw new RuntimeException("The instance id is "+instanceId);
		this._instanceId = instanceId;
		this._weight = weight;
	}
	
	public void setInstanceId(int instanceId){
		this._instanceId = instanceId;
	}
	
	public int getInstanceId(){
		return this._instanceId;
	}
	
	public double getWeight(){
		return this._weight;
	}
	
	public void setWeight(double weight){
		this._weight = weight;
	}
	
	/**
	 * The size of this instance, usually the length of the input sequence
	 * @return
	 */
	public abstract int size();
	
	public boolean isLabeled(){
		return this._isLabeled;
	}
	
	/**
	 * Set this instance as a labeled instance
	 */
	public void setLabeled(){
//		if(this.getOutput()==null){
//			throw new RuntimeException("This instance has no outputs, but you want to make it labeled??");
//		}
		this._isLabeled = true;
	}
	
	/**
	 * Set this instance as an unlabeled instance
	 */
	public void setUnlabeled(){
		this._isLabeled = false;
	}
	
	/**
	 * Return the duplicate (i.e., clone) of the current instance
	 * @return
	 */
	public abstract Instance duplicate();
	
	public abstract void removeOutput();
	public abstract void removePrediction();
	
	public abstract Object getInput();
	public abstract Object getOutput();
	public abstract Object getPrediction();
	
	public abstract boolean hasOutput();
	public abstract boolean hasPrediction();
	
	public abstract void setPrediction(Object o);
	
}