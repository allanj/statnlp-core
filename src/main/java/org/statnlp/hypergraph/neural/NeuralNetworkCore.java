package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.*;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkConfig.ModelStatus;

import edu.cmu.dynet.internal.ComputationGraph;
import edu.cmu.dynet.internal.Dim;
import edu.cmu.dynet.internal.Expression;
import edu.cmu.dynet.internal.FloatVector;
import edu.cmu.dynet.internal.LongVector;
import edu.cmu.dynet.internal.LookupParameterStorage;
import edu.cmu.dynet.internal.LookupParameterStorageVector;
import edu.cmu.dynet.internal.ParameterCollection;
import edu.cmu.dynet.internal.ParameterStorage;
import edu.cmu.dynet.internal.ParameterStorageVector;
import edu.cmu.dynet.internal.Tensor;
import edu.cmu.dynet.internal.TensorTools;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

public abstract class NeuralNetworkCore extends AbstractNeuralNetwork implements Cloneable {
	
	private static final long serialVersionUID = -2638896619016178432L;

	protected transient boolean isTraining;
	
	protected transient ParameterCollection modelParams;
	
	protected transient int numParams;
	
	protected float[] params;
	
	protected transient Expression currExpr;
	
//	protected transient FloatVector outputVector;
	/**
	 * Neural network input to index (id)
	 * If you are using batch training, do not directly use this to obtain input id.
	 * Use the method # {@link #getNNInputID()}
	 */
	protected transient Map<Object, Integer> nnInput2Id;
	
	protected transient List<Object> nnInputs;
	
	/**
	 * Save the mapping from instance id to neural network input id.
	 */
	protected transient TIntObjectMap<TIntList> instId2NNInputId;
	
	protected transient TIntIntMap dynamicNNInputId2BatchInputId;
	
	protected boolean continuousFeatureValue = false;
	
	protected String nnModelFile = null;
	
	public NeuralNetworkCore(int numLabels) {
		super(numLabels); //using cpu for default
		
	}
	
	public abstract ParameterCollection initalizeModelParams();
	
	public abstract Expression buildForwardGraph(List<Object> inputs);
	
	private void getNumParameters() {
		this.numParams = 0;
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			this.numParams += lps.size();
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			this.numParams += ps.size();
		}
		System.out.println("number of NN parameters: " + this.numParams);
	}
	
	public void initializeInput() {
		System.out.println("initializing");
		if (this.isTraining) {
			this.modelParams = this.initalizeModelParams();
			this.getNumParameters();
			this.gradParams = new float[this.numParams];
			this.params = new float[this.numParams];
			Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
			for (int i = 0; i < this.params.length; i++) {
				this.params[i] = (float) (NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextFloat()-0.5)/10 : NetworkConfig.FEATURE_INIT_WEIGHT);
			}
			System.out.println("finish initialize the nn parameters in Java side");
		}
		nnInputs = new ArrayList<>(nnInput2Id.size());
		for (Object obj : nnInput2Id.keySet()) {
			nnInputs.add(obj);
		}
		System.out.println("finish the initialization");
	}
	
	/**
	 * Calculate the input position in the output/countOuput matrix position
	 * @return
	 */
	public abstract int hyperEdgeInput2OutputRowIndex(Object edgeInput);
	
	public int getNNInputID(Object nnInput) {
		if (NetworkConfig.USE_BATCH_TRAINING && isTraining) {
			return this.dynamicNNInputId2BatchInputId.get(this.nnInput2Id.get(nnInput));
		} else {
			return this.nnInput2Id.get(nnInput);
		}
	}
	
	public int getNNInputSize() {
		if (NetworkConfig.USE_BATCH_TRAINING && isTraining) {
			return this.dynamicNNInputId2BatchInputId.size();
		} else {
			return this.nnInput2Id.size();
		}
	}
	
	/**
	 * Neural network's forward
	 */
	@Override
	public void forward(TIntSet batchInstIds) {
		
		if ( isTraining || NetworkConfig.STATUS == ModelStatus.TESTING
				|| NetworkConfig.STATUS == ModelStatus.DEV_IN_TRAINING) { // update with new params
			//copy the parameters
			this.copyParams();
		}
		
		if (NetworkConfig.USE_BATCH_TRAINING && isTraining && batchInstIds != null
				&& batchInstIds.size() > 0) {
			//pass the batch input id.
			TIntIterator iter = batchInstIds.iterator();
			TIntHashSet set = new TIntHashSet();
			while(iter.hasNext()) {
				int positiveInstId = iter.next();
				if (this.instId2NNInputId.containsKey(positiveInstId))
					set.addAll(this.instId2NNInputId.get(positiveInstId));
			}
			TIntList batchInputIds = new TIntArrayList(set);
			this.dynamicNNInputId2BatchInputId = new TIntIntHashMap(batchInputIds.size());
			for (int i = 0; i < batchInputIds.size(); i++) {
				this.dynamicNNInputId2BatchInputId.put(batchInputIds.get(i), i);
			}
			this.cg = ComputationGraph.getNew();
			List<Object> batchInputs = new ArrayList<>(batchInputIds.size());
			for (int i = 0; i < batchInputIds.size(); i++) {
				batchInputs.add(nnInputs.get(batchInputIds.get(i)));
			}
//			System.out.println("batch size: " + batchInputs.size());
//			this.currExpr = this.buildForwardGraph(nnInputs.get(batchInputIds.get(0)));
			this.currExpr = this.buildForwardGraph(batchInputs);
			Tensor outputTensor = this.cg.forward(this.currExpr);
			FloatVector outputVector = as_vector(outputTensor);
			this.output = this.getArray(outputVector, this.output);
			if (isTraining && (this.gradOutput == null || this.gradOutput.length < this.output.length)) {
				this.gradOutput = new float[this.output.length];
			}
		} else {
			this.cg = ComputationGraph.getNew();
			System.out.println("nninput size: " + nnInputs.size());
			this.currExpr = this.buildForwardGraph(nnInputs);
			Tensor outputTensor = this.cg.forward(this.currExpr);
			FloatVector outputVector = as_vector(outputTensor);
			this.output = this.getArray(outputVector, this.output);
		}
	}
	
	/**
	 * Copy the parameters from our framework to dynet.
	 */
	private void copyParams() {
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		int k = 0;
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			Tensor vals = lps.get_all_values();
			for(int i = 0; i < lps.size(); i++) {
				TensorTools.set_element(vals, i, this.params[k++]);
			}
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			Tensor vals = ps.getValues();
			for(int i = 0; i < ps.size(); i++) {
				TensorTools.set_element(vals, i, this.params[k++]);
			}
		}
	}
	
	private void copyGradParams() {
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		int k = 0;
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			Tensor vals = lps.get_all_grads();
			for(int i = 0; i < lps.size(); i++) {
				TensorTools.set_element(vals, i, this.gradParams[k++]);
			}
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			Tensor vals = ps.gradients();
			for(int i = 0; i < ps.size(); i++) {
				TensorTools.set_element(vals, i, this.gradParams[k++]);
			}
		}
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index) {
		double val = 0.0;
		NeuralIO io = getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getInput();
			int outputLabel = io.getOutput();
			int idx = this.hyperEdgeInput2OutputRowIndex(edgeInput) * this.numOutputs + outputLabel;
			val = output[idx];
		}
		return val;
	}
	
	static Dim makeDim(int[] dims) {
	    LongVector dimInts = new LongVector();
	    for (int i = 0; i < dims.length; i++) {
	      dimInts.add(dims[i]);
	    }
	    return new Dim(dimInts);
	}
	
	/**
	 * Neural network's backpropagation
	 */
	@Override
	public void backward() {
		
		this.zeroGradCG();
		FloatVector gradVec = this.getVector(this.gradOutput);
//		Dim currExprDim = this.currExpr.dim();
//		int[] dims = new int[1 + (int)currExprDim.ndims()];
//		int x = this.gradOutput.length;
//		for (int i = 0; i < currExprDim.ndims(); i++) {
//			dims[i] = (int)currExprDim.get(i);
//			x /= dims[i];
//		}
//		dims[dims.length - 1] = x;
		Expression myGrad = input(cg, this.currExpr.dim(), gradVec);
		Expression finalLoss = sum_elems(sum_batches(cmult(this.currExpr, myGrad)));
		this.cg.incremental_forward(finalLoss);
		this.cg.backward(finalLoss);
		this.copyGradParams();
		if (NetworkConfig.REGULARIZE_NEURAL_FEATURES) {
			addL2ParamsGrad();
		}
		this.resetCountOutput();
	}
	
	@Override
	public void update(double count, Network network, int parent_k, int children_k_index) {
		NeuralIO io = getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getInput();
			int outputLabel = io.getOutput();
			int idx = this.hyperEdgeInput2OutputRowIndex(edgeInput) * this.numOutputs + outputLabel;
			synchronized (gradOutput) {
				//TODO: alternatively, create #threads of countOutput array.
				//Then aggregate them together.
				gradOutput[idx] -= count;
			}
		}
	}
	
	private void zeroGradCG() {
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			TensorTools.zero(lps.get_all_grads());
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			TensorTools.zero(ps.gradients());
		}
	}
	
	public void resetCountOutput() {
		Arrays.fill(gradOutput, (float)0.0);
	}
	
	/**
	 * Read a DoubleTensor to a buffer.
	 * @param t
	 * @param buf
	 * @return
	 */
	protected float[] getArray(FloatVector t, float[] buf) {
		if (buf == null || buf.length < t.size()) {
			buf = new float[(int)t.size()];
        }
		for (int i = 0; i < t.size(); i++) {
			buf[i] = t.get(i);
		}
		return buf;
	}
	
	protected FloatVector getVector(float[] buf) {
		FloatVector vec = new FloatVector(buf.length);
		for (int i = 0; i < buf.length; i++) {
			vec.add(buf[i]);
		}
		return vec;
	}

	@Override
	protected NeuralNetworkCore clone(){
		NeuralNetworkCore c = null;
		try {
			c = (NeuralNetworkCore) super.clone();
			c.nnInput2Id = null;
			c.nnInputs = null;
			c.params = this.params;
			c.modelParams = this.modelParams;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return c;
	}
	
	public NeuralNetworkCore setModelFile(String nnModelFile) {
		this.nnModelFile = nnModelFile;
		return this;
	}

	private void writeObject(ObjectOutputStream out) throws IOException{
		out.writeBoolean(this.continuousFeatureValue);
		out.writeInt(this.netId);
		out.writeInt(this.numOutputs);
		out.writeDouble(this.scale);
		out.writeObject(this.nnModelFile);
	}
	
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException{
		this.continuousFeatureValue = in.readBoolean();
		this.netId = in.readInt();
		this.numOutputs = in.readInt();
		this.scale = in.readDouble();
		this.nnModelFile = (String) in.readObject();
	}
	
}


