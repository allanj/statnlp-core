
package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.initialize;

import java.io.Serializable;
import java.util.Arrays;

import org.statnlp.commons.ml.opt.MathsVector;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;

import edu.cmu.dynet.internal.ComputationGraph;
import edu.cmu.dynet.internal.DynetParams;
import gnu.trove.set.TIntSet;


public abstract class AbstractNeuralNetwork implements Serializable{
	
	private static final long serialVersionUID = 1501009887917654699L;

	/**
	 * The total number of unique outputs
	 */
	protected int numOutputs;
	
	/**
	 * The neural net's internal weights and gradients
	 */
	protected transient double[] params, gradParams;
	
	/**
	 * A flattened matrix containing the continuous values
	 * with the shape (inputSize x numLabels).
	 */
	protected transient float[] output, gradOutput;
	
	/**
	 * The coefficient used for regularization, i.e., batchSize/totalInstNum.
	 */
	protected double scale;
	
//	protected transient LocalNetworkParam[] params_l;
	
	protected transient DynetParams dp;
	protected transient ComputationGraph cg;
	
	/**
	 * Initialize the network with the number of outputs
	 * @param numLabels
	 */
	public AbstractNeuralNetwork(int numLabels) {
		this(numLabels, 1234);
	}
	
	/**
	 * Constructor
	 * @param numLabels
	 * @param randomSeed
	 */
	public AbstractNeuralNetwork(int numLabels, long randomSeed) {
		this.numOutputs = numLabels;
		this.dp = new DynetParams();
		dp.setRandom_seed(randomSeed);
		initialize(dp);
		this.cg = ComputationGraph.getNew();
	}
	
//	public void setLocalNetworkParams (LocalNetworkParam[] params_l) {
//		this.params_l = params_l;
//	}
	
	public abstract Object hyperEdgeInput2NNInput(Object edgeInput);
	
	/**
	 * Initialize this provider (e.g., create a network and prepare its input)
	 */
	public abstract void initModelParameters();
	
	/**
	 * Get the score associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return score
	 */
	public abstract double getScore(Network network, int parent_k, int children_k_index, NNDataHelper helper);
	
	/**
	 * Pre-compute all scores for each hyper-edge.
	 * In neural network, this is equivalent to forward.
	 */
	public abstract void forward(TIntSet batchInstIds, Object[] allInputs);
	
	/**
	 * Accumulate count for a specified hyper-edge
	 * @param count
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 */
	public abstract void update(double count, Network network, int parent_k, int children_k_index, NNDataHelper helper);
	
	/**
	 * Compute gradient based on the accumulated counts from all hyper-edges.
	 * In neural network, this is equivalent to backward.
	 */
	public abstract void backward();
	
	/**
	 * Get the input associated with a specified hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 * @return input
	 */
	
	
	/**
	 * Reset gradient
	 */
	public void resetGrad() {
		if (gradOutput != null) {
			Arrays.fill(gradOutput, (float)0.0);
		}
		if (gradParams != null && getParamSize() > 0) {
			Arrays.fill(gradParams, (float)0.0);
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
	
	public ComputationGraph getGraph(){
		return this.cg;
	}
	
}
