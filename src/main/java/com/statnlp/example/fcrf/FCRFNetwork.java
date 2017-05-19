package com.statnlp.example.fcrf;

import java.util.Arrays;

import com.statnlp.example.fcrf.FCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.InferenceType;
import com.statnlp.hybridnetworks.TableLookupNetwork;

public class FCRFNetwork extends TableLookupNetwork{

	private static final long serialVersionUID = -5035676335489326537L;

	int _numNodes = -1;
	
	int structure; 
	
	public FCRFNetwork(){
		
	}
	
	public FCRFNetwork(int networkId, FCRFInstance inst, LocalNetworkParam param){
		super(networkId, inst, param);
	}
	
	public FCRFNetwork(int networkId, FCRFInstance inst, long[] nodes, int[][][] children, LocalNetworkParam param, int numNodes){
		super(networkId, inst,nodes, children, param);
		this._numNodes = numNodes;
		this.isVisible = new boolean[nodes.length];
		if (NetworkConfig.INFERENCE == InferenceType.MEAN_FIELD)
			this.structArr = new int[nodes.length];
		Arrays.fill(isVisible, true);
	}
	
	public int countNodes(){
		if(this._numNodes==-1)
			return super.countNodes();
		else return this._numNodes;
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
