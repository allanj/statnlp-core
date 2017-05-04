package com.statnlp.example.mention_hypergraph;

import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.TableLookupNetwork;

public class MentionHypergraphNetwork extends TableLookupNetwork {

	private static final long serialVersionUID = 7173683038115335356L;
	
	public int numNodes = -1;

	public MentionHypergraphNetwork() {}

	public MentionHypergraphNetwork(int networkId, Instance inst, LocalNetworkParam param) {
		super(networkId, inst, param);
	}

	public MentionHypergraphNetwork(int networkId, Instance inst, long[] nodes, int[][][] children, LocalNetworkParam param, int numNodes) {
		super(networkId, inst, nodes, children, param);
		this.numNodes = numNodes;
		this.isVisible = new boolean[numNodes];
		for(int i=0; i<isVisible.length; i++){
			this.isVisible[i] = true;
		}
	}
	
	public int countNodes(){
		if(numNodes < 0){
			return super.countNodes();
		}
		return numNodes;
	}

}
