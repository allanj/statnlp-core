package com.statnlp.neural;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

import com.statnlp.hybridnetworks.FeatureValueProvider;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.neural.util.LuaFunctionHelper;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;
import th4j.Tensor.DoubleTensor;

public abstract class NeuralNetworkFeatureValueProvider extends FeatureValueProvider {
	
	protected HashMap<String,Object> config;
	
	/**
	 * Corresponding Torch tensors for params and gradParams
	 */
	protected DoubleTensor paramsTensor, gradParamsTensor;
	
	/**
	 * Corresponding Torch tensors for output and gradOutput
	 */
	protected DoubleTensor outputTensorBuffer, countOutputTensorBuffer;
	
	public boolean optimizeNeural;
	
	/**
	 * The current batch FVP input ids.
	 */
	private TIntArrayList currBatchFVPInputIds;
	
	public NeuralNetworkFeatureValueProvider(int numLabels) {
		super(numLabels);
		config = new HashMap<>();
		optimizeNeural = NetworkConfig.OPTIMIZE_NEURAL;
		currBatchFVPInputIds = new TIntArrayList();
		config.put("optimizeNeural", optimizeNeural);
	}
	
	@Override
	public void initialize() {
		int inputSize = this.getNNInputSize();
		this.initializeNN(inputSize);
		if (this.isTraining && NetworkConfig.USE_BATCH_TRAINING) {
			this.makeInstsId2FVPInputId();
		}
	}
	
	/**
	 * Initialize the neural network with specific inputSize.
	 * @param inputSize
	 */
	private void initializeNN(int inputSize) {
		if (isTraining) {
			this.countOutput = new double[inputSize * this.numLabels];
			// Pointer to Torch tensors
	        this.outputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
	        this.countOutputTensorBuffer = new DoubleTensor(inputSize, this.numLabels);
		}
		
		// Forward matrices
        this.output = new double[inputSize * this.numLabels];
		
		config.put("isTraining", isTraining);
        Object[] args = new Object[]{config, this.outputTensorBuffer, this.countOutputTensorBuffer};
        Class<?>[] retTypes;
        if (optimizeNeural && isTraining) {
        	retTypes = new Class[]{DoubleTensor.class, DoubleTensor.class};
        } else {
        	retTypes = new Class[]{};
        }
        Object[] outputs = LuaFunctionHelper.execLuaFunction(this.L, "initialize", args, retTypes);
        
		if(optimizeNeural && isTraining) {
			this.paramsTensor = (DoubleTensor) outputs[0];
			this.gradParamsTensor = (DoubleTensor) outputs[1];
			if (this.paramsTensor.nElement() > 0) {
				this.params = this.getArray(this.paramsTensor, this.params);
				//TODO: this one might not be needed. Because the gradient at the first initialization is 0..
				this.gradParams = this.getArray(this.gradParamsTensor, this.gradParams);
				if (NetworkConfig.INIT_FV_WEIGHTS) {
					Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
					//also be careful that you may overwrite the initialized embedding if you use this.
					for(int i = 0; i < this.params.length; i++) {
						this.params[i] = NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextDouble()-.5)/10 :
							NetworkConfig.FEATURE_INIT_WEIGHT;
					}
				}
			}
		}
	}
	
	/**
	 * Return the input size 
	 * @return input size of the neural net
	 */
	public abstract int getNNInputSize();
	
	/**
	 * Make sure the fvpInput2Id is now fixed.
	 */
	private void makeInstsId2FVPInputId() {
		for (int instanceId : this.instId2FVPInput.keys()) {
			Set<Object> fvpInputs = this.instId2FVPInput.get(instanceId);
			TIntList fvpInputIds = new TIntArrayList();
			for (Object fvpInput : fvpInputs) {
				fvpInputIds.add(fvpInput2id.get(fvpInput));
			}
			this.instId2FVPInputId.put(instanceId, fvpInputIds);
		}
		//System.out.println("inst id to fvp input id: " + this.instId2FVPInputId.toString());
		this.instId2FVPInput = null;
	}
	
	@Override
	public void initializeScores() {
		forward();
	}
	
	/**
	 * Calculate the input position in the output/countOuput matrix position
	 * @return
	 */
	public abstract int input2Index(Object input);
	
	@Override
	public void update() {
		backward();
	}
	
	/**
	 * Neural network's forward
	 */
	public void forward() {
		if (optimizeNeural) { // update with new params
			if (getParamSize() > 0) {
				this.paramsTensor.storage().copy(this.params); // we can do this because params is contiguous
				//System.out.println("java side forward weights: " + this.params[0] + " " + this.params[1]);
			}
		}
		Object[] args = null; 
		if (isTraining && NetworkConfig.USE_BATCH_TRAINING) {
			TIntIterator iter = this.batchInstIds.iterator();
			TIntHashSet set = new TIntHashSet();
			while(iter.hasNext()) {
				int positiveInstId = iter.next();
				assert(positiveInstId > 0);
				set.addAll(this.instId2FVPInputId.get(positiveInstId));
			}
			currBatchFVPInputIds = new TIntArrayList(set);
			args = new Object[]{isTraining, this.currBatchFVPInputIds};
		} else {
			args = new Object[]{isTraining};
		}
		Class<?>[] retTypes = new Class[]{DoubleTensor.class};
		LuaFunctionHelper.execLuaFunction(this.L, "forward", args, retTypes);
		output = this.getArray(outputTensorBuffer, output);
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		SimpleImmutableEntry<Object, Integer> io = getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getKey();
			int outputLabel = io.getValue();
			int idx = this.input2Index(edgeInput) * this.numLabels + outputLabel;
			val = output[idx];
		}
		return val;
	}
	
	/**
	 * Neural network's backpropagation
	 */
	public void backward() {
		countOutputTensorBuffer.storage().copy(this.countOutput);
		Object[] args = new Object[]{};
		Class<?>[] retTypes = new Class[0];
		LuaFunctionHelper.execLuaFunction(this.L, "backward", args, retTypes);
		
		if(optimizeNeural && getParamSize() > 0) { // copy gradParams computed by Torch
			gradParams = this.getArray(this.gradParamsTensor, gradParams);
			if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
				addL2ParamsGrad();
			}
		}
		this.resetCountOutput();
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		SimpleImmutableEntry<Object, Integer> io = getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getKey();
			int outputLabel = io.getValue();
			int idx = this.input2Index(edgeInput) * this.numLabels + outputLabel;
			synchronized (countOutput) {
				countOutput[idx] -= count;
			}
		}
	}
	
	public void resetCountOutput() {
		Arrays.fill(countOutput, 0.0);
	}
	
	/**
	 * Save the model by calling the specific function in Torch
	 * @param func : the function in torch
	 * @param prefix : model prefix
	 */
	public void save(String func, String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, func, new Object[]{prefix}, new Class[]{});
	}
	
	/**
	 * Save the trained model, implement the "save_model" method in torch
	 * @param prefix
	 */
	public void save(String prefix) {
		this.save("save_model", prefix);
	}
	
	/**
	 * Load a trained model, using the specific function in Torch
	 * @param func: the specific function for loading model
	 * @param prefix: model prefix.
	 */
	public void load(String func, String prefix) {
		LuaFunctionHelper.execLuaFunction(this.L, func, new Object[]{prefix}, new Class[]{});
	}
	
	/**
	 * Load a model from disk, implement the "load_model" method in torch
	 * @param prefix
	 */
	public void load(String prefix) {
		this.load("load_model", prefix);
	}
	
	@Override
	public void closeProvider() {
		this.cleanUp();
	}
	
	/**
	 * Clean up resources, currently, we clean up the resource after decoding
	 */
	public void cleanUp() {
		L.close();
	}
	
	/**
	 * Read a DoubleTensor to a buffer.
	 * @param t
	 * @param buf
	 * @return
	 */
	protected double[] getArray(DoubleTensor t, double[] buf) {
		if (buf == null || buf.length != t.nElement()) {
			buf = new double[(int) t.nElement()];
        }
		t.storage().getRawData().read(0, buf, 0, (int) t.nElement());
		return buf;
	}
}
