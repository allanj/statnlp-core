package com.statnlp.hybridnetworks;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Set;

import com.naef.jnlua.LuaState;
import com.statnlp.commons.ml.opt.MathsVector;
import com.sun.jna.Library;
import com.sun.jna.Native;

import gnu.trove.list.TIntList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;


public abstract class FeatureValueProvider {
	
	public static String LUA_VERSION = "5.2";
	
	/**
	 * A LuaState instance for loading Lua scripts
	 */
	public LuaState L;
	
	/**
	 * The total number of unique output labels
	 */
	protected int numLabels;
	
	/**
	 * The provider's internal weights and gradients
	 */
	protected double[] params, gradParams;
	
	/**
	 * A flattened matrix containing the continuous values
	 * with the shape (inputSize x numLabels).
	 */
	protected transient double[] output, countOutput;
	
	/**
	 * Maps a hyper-edge tuple (instance ID,parent node ID,child index) to an input-output pair
	 */
	protected TIntObjectHashMap<TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>> edge2io;
	
	/**
	 * Maps an feature value provider input to a corresponding index
	 */
	protected LinkedHashMap<Object,Integer> fvpInput2id;
	
	/**
	 * The coefficient used for regularization, i.e., batchSize/totalInstNum.
	 */
	protected double scale;
	
	/**
	 * The map for mapping the instance id to feature value provider input.
	 * Used for batch training
	 * TODO: Need to remove this later after we have multi-threaded {@link #addHyperEdge(Network, int, int, Object, int)}.
	 */
	protected TIntObjectMap<Set<Object>> instId2FVPInput;
	
	/**
	 * The map for mapping the instance id to feature value provider input id.
	 * Used for batch training.
	 */
	protected TIntObjectMap<TIntList> instId2FVPInputId;
	
	/**
	 * The batchInstIds obtained from NetworkModel class
	 */
	protected transient TIntSet batchInstIds;
	
	/**
	 * Whether we are training or decoding
	 */
	protected boolean isTraining = true;
	
	public FeatureValueProvider(int numLabels) {
		edge2io = new TIntObjectHashMap<TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>>();
		fvpInput2id = new LinkedHashMap<Object,Integer>();
		instId2FVPInput = new TIntObjectHashMap<>();
		instId2FVPInputId = new TIntObjectHashMap<>();
		this.numLabels = numLabels;
		this.configureJNLua();
	}
	
	/**
	 * Configure paths for JNLua and create a new LuaState instance
	 * for loading the backend Torch/Lua script
	 */
	private void configureJNLua() {
		System.setProperty("jna.library.path","./nativeLib");
		System.setProperty("java.library.path", "./nativeLib:" + System.getProperty("java.library.path"));
		Field fieldSysPath = null;
		try {
			fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
			fieldSysPath.setAccessible(true);
			fieldSysPath.set(null, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
		String jnluaLib = null;
		if (LUA_VERSION.equals("5.2")) {
			jnluaLib = "libjnlua52";
		} else if (LUA_VERSION.equals("5.1")) {
			jnluaLib = "libjnlua5.1";
		}
		if (NetworkConfig.OS.equals("osx")) {
			jnluaLib += ".jnilib";
		} else if (NetworkConfig.OS.equals("linux")) {
			jnluaLib += ".so";
		}
		Native.loadLibrary(jnluaLib, Library.class);
		
		this.L = new LuaState();
		this.L.openLibs();
		
		try {
			this.L.load(Files.newInputStream(Paths.get("nn-crf-interface/neural_server/NetworkInterface.lua")),"NetworkInterface.lua","bt");
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.L.call(0,0);
	}
	
	/**
	 * Add a hyper-edge and its corresponding input-output pair
	 * TODO: make it multi-threaded
	 * TODO: replace {@link #instId2FVPInput} with {@link #instId2FVPInputId} directly.   
	 * @param network
	 * @param parent_k
	 * @param children_k_idx
	 * @param input
	 * @param output
	 */
	public synchronized void addHyperEdge(Network network, int parent_k, int children_k_idx, Object edgeInput, int output) {
		int instanceID = network.getInstance().getInstanceId();
		if ( ! edge2io.containsKey(instanceID)) {
			edge2io.put(instanceID, new TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>>());
		}
		if ( ! edge2io.get(instanceID).containsKey(parent_k)) {
			edge2io.get(instanceID).put(parent_k, new TIntObjectHashMap<SimpleImmutableEntry<Object,Integer>>());
		}
		if ( ! edge2io.get(instanceID).get(parent_k).containsKey(children_k_idx)) {
			edge2io.get(instanceID).get(parent_k).put(children_k_idx, new SimpleImmutableEntry<Object,Integer>(edgeInput, output));
		}
		Object fvpInput = edgeInput2FVPInput(edgeInput);
		if ( ! fvpInput2id.containsKey(fvpInput)) {
			fvpInput2id.put(fvpInput, fvpInput2id.size());
		}
		if (isTraining && NetworkConfig.USE_BATCH_TRAINING) {
			//TODO: we can actually add those instance id with one sign (e.g. positive)
			if (! instId2FVPInput.containsKey(Math.abs(instanceID))) {
				Set<Object> set = new HashSet<>();
				set.add(fvpInput);
				instId2FVPInput.put(instanceID, set);
			} else {
				instId2FVPInput.get(instanceID).add(fvpInput);
			}
		}
	}
	
	public abstract Object edgeInput2FVPInput(Object edgeInput);
	
	/**
	 * Initialize this provider (e.g., create a network and prepare its input)
	 */
	public abstract void initialize();
	
	/**
	 * Get the score associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return score
	 */
	public abstract double getScore(Network network, int parent_k, int children_k_index);
	
	/**
	 * Pre-compute all scores for each hyper-edge.
	 * In neural network, this is equivalent to forward.
	 */
	public abstract void initializeScores();
	
	/**
	 * Accumulate count for a specified hyper-edge
	 * @param count
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 */
	public abstract void update(double count, Network network, int parent_k, int children_k_index);
	
	/**
	 * Compute gradient based on the accumulated counts from all hyper-edges.
	 * In neural network, this is equivalent to backward.
	 */
	public abstract void update();
	
	/**
	 * Get the input associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return input
	 */
	public SimpleImmutableEntry<Object, Integer> getHyperEdgeInputOutput(Network network, int parent_k, int children_k_index) {
		int instanceID = network.getInstance().getInstanceId();
		TIntObjectHashMap<TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>>> tmp = edge2io.get(instanceID);
		if (tmp == null)
			return null;
		
		TIntObjectHashMap<SimpleImmutableEntry<Object, Integer>> tmp2 = tmp.get(parent_k);
		if (tmp2 == null)
			return null;
		
		return tmp2.get(children_k_index);
	}
	
	/**
	 * Call this method after the training is finished
	 * Do not call this if it's during training.
	 */
	public void clearInputAndEdgeMapping() {
		fvpInput2id.clear();
		edge2io.clear();
		if (NetworkConfig.USE_BATCH_TRAINING) {
			instId2FVPInputId = null;
		}
	}
	
	public abstract void closeProvider();
	
	/**
	 * Reset gradient
	 */
	public void resetGrad() {
		if (countOutput != null) {
			Arrays.fill(countOutput, 0.0);
		}
		if (gradParams != null && getParamSize() > 0) {
			Arrays.fill(gradParams, 0.0);
		}
	}
	
	public double getL2Params() {
		if (getParamSize() > 0) {
			return MathsVector.square(params);
		}
		return 0.0;
	}
	
	public void addL2ParamsGrad() {
		if (getParamSize() > 0) {
			double _kappa = NetworkConfig.L2_REGULARIZATION_CONSTANT;
			for(int k = 0; k<gradParams.length; k++) {
				if(_kappa > 0) {
					gradParams[k] += 2 * scale * _kappa * params[k];
				}
			}
		}
	}
	
	public int getParamSize() {
		return params == null ? 0 : params.length;
	}

	public double[] getParams() {
		return params;
	}

	public double[] getGradParams() {
		return gradParams;
	}
	
	public void setScale(double coef) {
		scale = coef;
	}
	
	public void setTrainingState() {
		isTraining = true;
	}
	
	public void setDecodingState() {
		isTraining = false;
	}
	
	public boolean isTraining() {
		return isTraining;
	}
	
	public void setBatchInstIds (TIntSet batchInstIds) {
		this.batchInstIds = batchInstIds;
	}
}
