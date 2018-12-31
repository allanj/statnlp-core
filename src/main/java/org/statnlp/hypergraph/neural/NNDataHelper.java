package org.statnlp.hypergraph.neural;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;

public class NNDataHelper implements Serializable{

	private static final long serialVersionUID = 6065466652568298006L;
	
	protected transient LocalNetworkParam[] params_l;
	
	protected transient Map<Object,Integer> allNNInput2Id;
	protected transient Object[] nnInputs;
	
	public NNDataHelper() {
		allNNInput2Id = new HashMap<>();
	}
	
	public void setLocalNetworkParams(LocalNetworkParam[] params_l) {
		this.params_l = params_l;
	}
	
	public Object[] getNNInputs() {
		 return this.nnInputs;
	}
	
	public int getNNInputId(Object nnInput) {
		return this.allNNInput2Id.get(nnInput);
	}
	
	public int getNNInputSize() {
		return this.allNNInput2Id.size();
	}
	
	public NeuralIO getHyperEdgeInputOutput(Network network, int parent_k, int children_k_index) {
		return this.params_l[network.getThreadId()].getHyperEdgeIO(network, parent_k, children_k_index);
	}
	
	public void prepareInputIdAndInput() {
		Map<Object,Integer> input2id_list = allNNInput2Id;
		for (LocalNetworkParam param_l : params_l) {
			input2id_list.putAll(param_l.getLocalNNInput2Id());	
			param_l.setLocalNNInput2Id(null);
		}
		//System.out.println(allNNInput2Id.get(0).size());
		int inputId = 0;
		for (Object input : allNNInput2Id.keySet()) {
			this.allNNInput2Id.put(input, inputId);
			inputId++;
		}
		input2id_list = null;
		this.nnInputs = new Object[this.allNNInput2Id.size()];
		for (Object obj : this.allNNInput2Id.keySet()) {
			this.nnInputs[this.allNNInput2Id.get(obj)] = obj;
		}
		System.out.println("neural inputs: "  + Arrays.toString(this.nnInputs));
	}
	
//	/**
//	 * Used for batch training.
//	 */
//	public void prepareInstId2NNInputId() {
//		this.net.instId2NNInputId = new TIntObjectHashMap<>();
//		for (LocalNetworkParam param_l : params_l) {
//			TIntObjectMap<Set<Object>> instId2NNInput =  param_l.getLocalInstId2NNInput();
//			for (int instId : instId2NNInput.keys()) {
//				Set<Object> set = instId2NNInput.get(instId);
//				TIntList list = new TIntArrayList();
//				for (Object obj : set) {
//					list.add(this.net.nnInput2Id.get(obj));
//				}
//				if (this.net.instId2NNInputId.containsKey(instId)) {
//					throw new RuntimeException("should unique for each local param.");
//				} else {
//					this.net.instId2NNInputId.put(instId, list);
//				}
//			}
//		}
//	}
	
}
