package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import com.statnlp.commons.types.Instance;
import com.statnlp.neural.AbstractTensor;

/**
 * The class used by {@link FeatureArray} to store the list of feature indices and 
 * the cached score of the features associated with this list, as a time-saving mechanism.<br>
 * This can also be used to save memory usage by not allocating new FeatureBox with 
 * the same feature indices as the one that is already created, by storing a cache in a LocalNetworkParam object.
 */
public class FeatureBox implements Serializable {

	private static final long serialVersionUID = 1779316632297457057L;

	/** Feature index array */
	protected int[] _fs;
	/** The total score (weights*values) of the feature in the current _fs. */
	protected double _currScore;
	/** Feature value array */
	private double[] _fv;
	
	/** The time-saving mechanism, by not recomputing the score if the version is up-to-date. */
	protected int _version;
	/** For now this is used for Mean-Field implementation, to update the weights during MF internal iterations */
	protected boolean _alwaysChange = false;
	
	public FeatureBox(int[] fs) {
		this._fs = fs;
		this._version = -1; //the score is not calculated yet.
	}
	
	public FeatureBox(int[] fs, double[] fv) {
		this._fs = fs;
		this._fv = fv;
		this._version = -1; //the score is not calculated yet.
	}
	
	public void setFeatureValues(Instance instance, LocalNetworkParam param) {
		if (this._fv != null) {
			//System.err.println("Feature values are already initialized.");
			return;
		}
		List<AbstractTensor> tensorList = instance.getTensorList();
		this._fv = new double[this._fs.length];
		for (int i = 0; i < this._fv.length; i++) {
			double val;
			int f = this._fs[i];
			if (NetworkConfig.USE_NEURAL_FEATURES && param.isNeural(f)) {
				int[] pos = param.getNeuralLocation(f);
				val = tensorList.get(pos[0]).get(pos[1]); 
			} else {
				val = 1.0;
			}
			this._fv[i] = val;
		}
	}
	
	public int length() {
		return this._fs.length;
	}
	
	public int[] get() {
		return this._fs;
	}
	
	public int get(int pos) {
		return this._fs[pos];
	}
	
	public double[] getValue() {
		return this._fv;
	}
	
	public double getValue(int pos) {
		return this._fv[pos];
	}

	/**
	 * Use the map to cache the feature index array to save the memory.
	 * @param fs
	 * @param param
	 * @return
	 */
	public static FeatureBox getFeatureBox(int[] fs, double[] fv, LocalNetworkParam param){
		FeatureBox fb = new FeatureBox(fs, fv);
		if (!NetworkConfig.AVOID_DUPLICATE_FEATURES) {
			return fb;
		}
		if (param.fbMap == null) {
			param.fbMap = new HashMap<FeatureBox, FeatureBox>();
		}
		if (param.fbMap.containsKey(fb)) {
			return param.fbMap.get(fb);
		} else{
			param.fbMap.put(fb, fb);
			return fb;
		}
	}
	
	public static FeatureBox getFeatureBox(int[] fs, LocalNetworkParam param){
		FeatureBox fb = new FeatureBox(fs);
		if (!NetworkConfig.AVOID_DUPLICATE_FEATURES) {
			return fb;
		}
		if (param.fbMap == null) {
			param.fbMap = new HashMap<FeatureBox, FeatureBox>();
		}
		if (param.fbMap.containsKey(fb)) {
			return param.fbMap.get(fb);
		} else{
			param.fbMap.put(fb, fb);
			return fb;
		}
	}
	
	@Override
	public int hashCode() {
		int hash = 1;
		int a = Arrays.hashCode(_fs);
		hash = hash * 17 + a;
		if (_fv != null) {
			int b = Arrays.hashCode(_fv);
			hash = hash * 31 + b;
		}
		return hash;
	}

	@Override
	public boolean equals(Object obj) {
		if(obj instanceof FeatureBox){
			FeatureBox other = (FeatureBox)obj;
			boolean isEqual = Arrays.equals(_fs, other._fs);
			if (_fv != null) {
				isEqual = isEqual && Arrays.equals(_fv, other._fv);
			} else {
				isEqual = isEqual && other._fv == null;
			}
			return isEqual;
		}
		return false;
	}
}
