/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.Arrays;
//import java.util.Map;
//import java.util.concurrent.ConcurrentHashMap;

import com.statnlp.commons.types.Instance;

/**
 * The base class for network compiler, a class to convert a problem representation between 
 * {@link Instance} (the surface form) and {@link Network} (the modeled form)<br>
 * When implementing the {@link #compile(int, Instance, LocalNetworkParam)} method, you might 
 * want to split the case into two cases: labeled and unlabeled, where the labeled network contains
 * only the existing nodes and edges in the instance, and the unlabeled network contains all
 * possible nodes and edges in the instance.
 * @author Wei Lu <luwei@statnlp.com>
 *
 */
public abstract class NetworkCompiler implements Serializable{
	
	private static final long serialVersionUID = 1052885626598299680L;
//	public Map<Integer, Double> instanceIDtoScore = new ConcurrentHashMap<Integer, Double>();
	
	/**
	 * Convert an instance into the network representation.<br>
	 * This process is also called the encoding part (e.g., to create the trellis network 
	 * of POS tags for a given sentence)<br>
	 * Subclasses might want to split this method into two, one for labeled instance, and 
	 * another for unlabeled instance.
	 * @param networkId
	 * @param inst
	 * @param param
	 * @return
	 */
	public abstract Network compile(int networkId, Instance inst, LocalNetworkParam param);
	
	/**
	 * Convert a network into an instance, the surface form.<br>
	 * This process is also called the decoding part (e.g., to get the sequence with maximum 
	 * probability in an HMM)
	 * @param network
	 * @return
	 */
	public abstract Instance decompile(Network network);

	
	/**
	 * The cost of the structure from leaf nodes up to node <code>k</code>.<br>
	 * This is used for structured SVM, and generally the implementation requires the labeled Instance.<br>
	 * This does some check whether the cost is actually required. Cost is not calculated during test, since
	 * there is no labeled instance.<br>
	 * This will call {@link #totalCostUpTo(int, int[])}, where the actual implementation resides.
	 * @param k
	 * @param child_k
	 * @return
	 */
	public double cost(Network network, int k, int[] child_k){
		if(network.getInstance().getInstanceId() > 0 || network.getInstance().isLabeled()){
			return 0.0;
		}
		return totalCostUpTo(network, k, child_k);
	}

	/**
	 * The cost of the structure from leaf nodes up to node <code>k</code>.<br>
	 * This is used for structured SVM, and generally the implementation requires the labeled Instance.<br>
	 * Note that the implementation can access the cost of the child nodes at {@link Network#getCost(int)} and
	 * the best path so far is stored at getMaxPath(child_idx), which represents the hyperedge connected to
	 * node <code>child_idx</code> which is part of the best path so far.
	 * @param k
	 * @param child_k
	 * @return
	 */	
	public double totalCostUpTo(Network network, int parent_k, int[] child_k){
		int size = network.getInstance().size();
		Network labeledNet = getLabeledNetwork(network);
		double aggregateChildLoss = aggregateChildCost(network, parent_k, child_k);
		long node = network.getNode(parent_k);
		int node_k = labeledNet.getNodeIndex(node);
		if(node_k < 0){
			return aggregateChildLoss;
		}
		long[] childNodes = new long[child_k.length];
		for(int i=0; i<child_k.length; i++){
			childNodes[i] = network.getNode(child_k[i]);
		}
		int[][] children_k = labeledNet.getChildren(node_k);
		boolean edgePresentInLabeled = false;
		for(int[] children: children_k){
			long[] childrenNodes = new long[children.length];
			for(int i=0; i<children.length; i++){
				childrenNodes[i] = labeledNet.getNode(children[i]);
			}
			if(Arrays.equals(childrenNodes, childNodes)){
				edgePresentInLabeled = true;
				break;
			}
		}
		if(edgePresentInLabeled){
			return aggregateChildLoss;
		} else {
			return aggregateChildLoss+NetworkConfig.SSVM_MARGIN/size;
		}
	}
	
	private Network getLabeledNetwork(Network network){
		Network labeledNet = network.getLabeledNetwork();
		if(labeledNet != null){
			return labeledNet;
		}
		Instance labeledInstance = network.getInstance().getLabeledInstance();
		if(labeledInstance != null){
			labeledNet = compile(-1, labeledInstance, new LocalNetworkParam(-1, null, 1));
			network.setLabeledNetwork(labeledNet);
			return labeledNet;
		} else {
			return null;
		}
	}
	
	/**
	 * Return the maximum cost value over all the child nodes for the specified network
	 * @param network
	 * @param k
	 * @param child_k
	 * @return
	 */
	public double aggregateChildCost(Network network, int k, int[] child_k){
		double maxLoss = 0.0;
		for(int child: child_k){
			maxLoss += network.getCost(child);
		}
		return maxLoss;
	}
	
}