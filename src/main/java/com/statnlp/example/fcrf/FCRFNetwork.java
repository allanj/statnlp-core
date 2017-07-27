package com.statnlp.example.fcrf;

import java.util.Arrays;

import com.statnlp.commons.types.Instance;
import com.statnlp.example.base.BaseNetwork;
import com.statnlp.example.fcrf.FCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hypergraph.LocalNetworkParam;
import com.statnlp.hypergraph.NetworkCompiler;
import com.statnlp.hypergraph.NetworkConfig;
import com.statnlp.hypergraph.NetworkConfig.InferenceType;

public class FCRFNetwork extends BaseNetwork {

	private static final long serialVersionUID = -5035676335489326537L;

	int structure; 
	
	public FCRFNetwork(){}
	
	public FCRFNetwork(int networkId, Instance inst, LocalNetworkParam param, NetworkCompiler compiler){
		super(networkId, inst, param, compiler);
	}
	
	public FCRFNetwork(int networkId, Instance inst, long[] nodes, int[][][] children, int numNodes, LocalNetworkParam param, NetworkCompiler compiler){
		super(networkId, inst, nodes, children, numNodes, param, compiler);
		this.isVisible = new boolean[nodes.length];
		if (NetworkConfig.INFERENCE == InferenceType.MEAN_FIELD)
			this.structArr = new int[nodes.length];
		Arrays.fill(isVisible, true);
	}
	
	public void remove(int k){
		this.isVisible[k] = false;
		if (this._inside != null){
			this._inside[k] = Double.NEGATIVE_INFINITY;
		}
		if (this._outside != null){
			this._outside[k] = Double.NEGATIVE_INFINITY;
		}
	}
	
	public boolean isRemoved(int k){
		return !this.isVisible[k];
	}
	
	public void recover(int k){
		this.isVisible[k] = true;
	}
	
	public void initStructArr() {
		for (int i = 0; i < this.countNodes(); i++) {
			int[] node_k = this.getNodeArray(i);
			if (node_k[2] == NODE_TYPES.LEAF.ordinal()) this.structArr[i] = 0;
			else if (node_k[2] == NODE_TYPES.ENODE.ordinal()) this.structArr[i] = 1;
			else if (node_k[2] == NODE_TYPES.TNODE.ordinal()) this.structArr[i] = 2;
			else if (node_k[2] == NODE_TYPES.ROOT.ordinal()) this.structArr[i] = 3;
			else throw new RuntimeException("unknown node type");
		}
	}
	
	/**
	 * 0 is the entity chain
	 * 1 is the PoS chain
	 */
	public void enableKthStructure(int kthStructure){
		if (kthStructure == 0) {
			// enable the chunking structure
			for (int i = 0; i < this.countNodes(); i++) {
				if (this.structArr[i] == 1 || this.structArr[i] == 0
						|| this.structArr[i] == 3)
					recover(i);
				else remove(i);
			}
		} else if (kthStructure == 1) {
			// enable POS tagging structure
			for (int i = 0; i < this.countNodes(); i++) {
				if (this.structArr[i] == 2 || this.structArr[i] == 0
						|| this.structArr[i] == 3)
					recover(i);
				else remove(i);
			}
		} else {
			throw new RuntimeException("removing unknown structures");
		}
	}
}
