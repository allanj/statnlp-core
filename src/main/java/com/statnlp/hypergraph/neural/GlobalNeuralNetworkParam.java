package com.statnlp.hypergraph.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.statnlp.hypergraph.LocalNetworkParam;
import com.statnlp.hypergraph.Network;

public class GlobalNeuralNetworkParam implements Serializable{

	private static final long serialVersionUID = 6065466652568298006L;

	protected List<NeuralNetworkCore> nets;
	
	protected LocalNetworkParam[] params_l;
	
	protected List<Map<Object,Integer>> allNNInput2Id;
	
	public GlobalNeuralNetworkParam() {
		this(new ArrayList<NeuralNetworkCore>());
	}
	
	public GlobalNeuralNetworkParam(List<NeuralNetworkCore> nets) {
		this.nets = nets;
		allNNInput2Id = new ArrayList<>();
		for (int id = 0; id < this.nets.size(); id++) {
			this.nets.get(id).setNeuralNetId(id);
			allNNInput2Id.add(new HashMap<>());
		}
	}
	
	/**
	 * Copy the global neural network 
	 * @return
	 */
	public GlobalNeuralNetworkParam copyNNParamG() {
		List<NeuralNetworkCore> newNets = new ArrayList<>(this.nets.size());
		for (int i = 0; i < this.nets.size(); i++) {
			newNets.add(this.nets.get(i).clone());
		}
		return new GlobalNeuralNetworkParam(newNets);
	}
	
	public void copyNNParam(GlobalNeuralNetworkParam src) {
		for (int id = 0; id < this.nets.size(); id++) {
			this.nets.get(id).params = src.getNet(id).params;
		}
	}
	
	public void setLocalNetworkParams(LocalNetworkParam[] params_l) {
		this.params_l = params_l;
		for (int id = 0; id < this.nets.size(); id++) {
			this.nets.get(id).setLocalNetworkParams(params_l);
		}
	}
	
	public void setLearningState() {
		for (int id = 0; id < this.nets.size(); id++) {
			this.nets.get(id).isTraining = true;
		}
	}
	
	public void setDecodeState() {
		for (int id = 0; id < this.nets.size(); id++) {
			this.nets.get(id).isTraining = false;
		}
	}
	
	/**
	 * Return all the neural network. 
	 * @return
	 */
	public List<NeuralNetworkCore> getAllNets() {
		return this.nets;
	}
	
	public NeuralNetworkCore getNet(int netId) {
		return this.nets.get(netId);
	}
	
	public void prepareInputId() {
		for (LocalNetworkParam param_l : params_l) {
			for (int netId = 0; netId < this.nets.size(); netId++) {
				allNNInput2Id.get(netId).putAll(param_l.getLocalNNInput2Id().get(netId));	
			}
		}
		System.out.println(allNNInput2Id.get(0).size());
		for (int netId = 0; netId < this.nets.size(); netId++) {
			this.nets.get(netId).nnInput2Id = new HashMap<Object, Integer>();
			int inputId = 0;
			for (Object input : allNNInput2Id.get(netId).keySet()) {
				this.nets.get(netId).nnInput2Id.put(input, inputId);
				inputId++;
			}
			allNNInput2Id.set(netId, null);
		}
		allNNInput2Id = null;
	}
	
	/**
	 * Building the neural network structure.
	 */
	public void initializeNetwork() {
		for (NeuralNetworkCore net : nets) {
			net.initialize();
		}
	}
	
	/**
	 * forward all the networks
	 */
	public void forward() {
		for (NeuralNetworkCore net : nets) {
			net.forward();
		}
	}
	
	/**
	 * Backpropagation.
	 */
	public void backward() {
		for (NeuralNetworkCore net : nets) {
			net.backward();
		}
	}
	
	/**
	 * Sum the provider scores for a given hyper-edge
	 * @param network
	 * @param parent_k
	 * @param children_k
	 * @param children_k_index
	 * @return
	 */
	public double getNNScore(Network network, int parent_k, int[] children_k, int children_k_index) {
		double score = 0.0;
		for (NeuralNetworkCore net : nets) {
			score += net.getScore(network, parent_k, children_k_index);
		}
		return score;
	}
	
	/**
	 * Send the count information for a given hyper-edge to each provider
	 * @param count
	 * @param network
	 * @param parent_k
	 * @param children_k_index
	 */
	public void setNNGradOutput(double count, Network network, int parent_k, int children_k_index) {
		for (NeuralNetworkCore net : nets) {
			net.update(count, network, parent_k, children_k_index);
		}
	}
	
	/**
	 * Close the Lua state connection
	 */
	public void closeNNConnections() {
		for (NeuralNetworkCore net : this.nets) {
			net.closeProvider();
		}
	}
	
	/**
	 * Reset accumulated gradient in each neural network
	 */
	public void resetAllNNGradients() {
		for (NeuralNetworkCore net : this.nets) {
			net.resetGrad();
		}
	}
	
}
