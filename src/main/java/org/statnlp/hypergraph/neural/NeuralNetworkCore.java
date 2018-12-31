package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.cmult;
import static edu.cmu.dynet.internal.dynet_swig.input;
import static edu.cmu.dynet.internal.dynet_swig.sum_batches;
import static edu.cmu.dynet.internal.dynet_swig.sum_elems;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
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
import gnu.trove.set.TIntSet;

public abstract class NeuralNetworkCore extends AbstractNeuralNetwork implements Cloneable {
	
	private static final long serialVersionUID = -2638896619016178432L;

	protected transient boolean isTraining;
	
	protected transient ParameterCollection modelParams;
	
	protected transient int numParams;
	
	protected transient Expression currExpr;
	
	protected transient FloatVector gradVec;
	
	protected String nnModelFile = null;
	
	public NeuralNetworkCore(int numLabels) {
		super(numLabels); //using cpu for default
		
	}
	
	public abstract ParameterCollection initSpecificModelParam();
	
	public abstract Expression buildForwardGraph(Object[] inputs);
	
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
	
	public ParameterCollection getModel(){
		return this.modelParams;
	}
	
	public void initModelParameters() {
		System.out.println("initializing neural network [only called during training]");
		this.modelParams = this.initSpecificModelParam();
		this.getNumParameters();
		this.gradParams = new double[this.numParams];
		this.params = new double[this.numParams];
		Random rng = new Random(NetworkConfig.RANDOM_INIT_FEATURE_SEED);
		for (int i = 0; i < this.params.length; i++) {
			this.params[i] = (float) (NetworkConfig.RANDOM_INIT_WEIGHT ? (rng.nextFloat()-0.5)/10 : NetworkConfig.FEATURE_INIT_WEIGHT);
		}
		System.out.println("finish initialize the nn parameters in Java side");
	}
	
	/**
	 * Calculate the input position in the output/countOuput matrix position
	 * @return
	 */
	public abstract int hyperEdgeInput2OutputRowIndex(Object edgeInput, NNDataHelper helper);
	
	/**
	 * Neural network's forward
	 */
	@Override
	public void forward(TIntSet batchInstIds, Object[] inputs) {
		if ( isTraining || NetworkConfig.STATUS == ModelStatus.TESTING
				|| NetworkConfig.STATUS == ModelStatus.DEV_IN_TRAINING) { // update with new params
			//copy the parameters
			/***
			if (NetworkConfig.STATUS == ModelStatus.TESTING)
				System.out.println("copying memory");
			***/
			this.copyParams();
			if (NetworkConfig.STATUS == ModelStatus.DEV_IN_TRAINING) {
				System.out.println("here");
			}
		}
		this.cg = ComputationGraph.getNew();
		this.currExpr = this.buildForwardGraph(inputs);
		Tensor outputTensor = this.cg.forward(this.currExpr);
//		FloatVector outputVector = as_vector(outputTensor); //try tensor tool access element because as_vector may cause memory increase
//		this.output = this.getArray(outputVector, this.output);
		this.output = this.copyOutput(outputTensor, this.output);
		if (isTraining && (this.gradOutput == null || this.gradOutput.length < this.output.length)) {
			this.gradOutput = new float[this.output.length];
		}
	}
	
	private float[] copyOutput(Tensor outputTensor, float[] buf) {
		int size = (int) (outputTensor.getD().rows() * outputTensor.getD().cols());
		if (buf == null || buf.length < size) {
			buf = new float[size];
        }
		for (int i = 0; i < size; i++) {
			buf[i] = TensorTools.access_element(outputTensor, i);
		}
		return buf;
	}
	
	/**
	 * Copy the parameters from our framework to dynet.
	 */
	private void copyParams() {
//		System.out.println("nn param java side : " + Arrays.toString(this.params));
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		int k = 0;
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			Tensor vals = lps.get_all_values();
			for(int i = 0; i < lps.size(); i++) {
				TensorTools.set_element(vals, i, (float)this.params[k++]);
			}
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			Tensor vals = ps.getValues();
			for(int i = 0; i < ps.size(); i++) {
				TensorTools.set_element(vals, i, (float)this.params[k++]);
			}
		}
	}
	
	/**
	 * Copy the gradients from dynet to our framework
	 */
	private void copyGradParams() {
		LookupParameterStorageVector lpsv =  modelParams.lookup_parameters_list();
		ParameterStorageVector psv =  modelParams.parameters_list();
		int k = 0;
		for (int l = 0; l < lpsv.size(); l++) {
			LookupParameterStorage lps = lpsv.get(l);
			Tensor vals = lps.get_all_grads();
			for(int i = 0; i < lps.size(); i++) {
				this.gradParams[k++] = TensorTools.access_element(vals, i);
			}
		}
		for (int l = 0; l < psv.size(); l++) {
			ParameterStorage ps = psv.get(l);
			Tensor vals = ps.gradients();
			for(int i = 0; i < ps.size(); i++) {
				this.gradParams[k++] = TensorTools.access_element(vals, i);
			}
		}
//		System.out.println("grad Param: " + Arrays.toString(this.gradParams));
	}
	
	@Override
	public double getScore(Network network, int parent_k, int children_k_index, NNDataHelper helper) {
		double val = 0.0;
//		if (!network.getInstance().isLabeled() && network.getInstance().getInstanceId() > 0) {
//			System.out.println(helper==null);
//		}
		NeuralIO io = helper.getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getInput();
			int outputLabel = io.getOutput();
			int idx = this.hyperEdgeInput2OutputRowIndex(edgeInput, helper) * this.numOutputs + outputLabel;
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
		this.getVector(this.gradOutput);
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
	public void update(double count, Network network, int parent_k, int children_k_index, NNDataHelper helper) {
		NeuralIO io = helper.getHyperEdgeInputOutput(network, parent_k, children_k_index);
		if (io != null) {
			Object edgeInput = io.getInput();
			int outputLabel = io.getOutput();
			int idx = this.hyperEdgeInput2OutputRowIndex(edgeInput, helper)* this.numOutputs + outputLabel;
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
	
	public boolean isLearningState() {
		return this.isTraining;
	}
	
	public void setLearningState() {
		this.isTraining = true;
	}
	
	public void setDecodeState() {
		this.isTraining = false;
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
	
	protected void getVector(float[] buf) {
		if (this.gradVec == null) {
			this.gradVec = new FloatVector(buf.length);
		}
		for (int i = 0; i < buf.length; i++) {
			this.gradVec.set(i, buf[i]);
		}
	}

	public NeuralNetworkCore setModelFile(String nnModelFile) {
		this.nnModelFile = nnModelFile;
		return this;
	}

	private void writeObject(ObjectOutputStream out) throws IOException{
		out.writeInt(this.numOutputs);
		out.writeDouble(this.scale);
		out.writeObject(this.nnModelFile);
	}
	
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException{
		this.numOutputs = in.readInt();
		this.scale = in.readDouble();
		this.nnModelFile = (String) in.readObject();
	}
	
}


