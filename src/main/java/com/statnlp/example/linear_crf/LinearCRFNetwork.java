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
/**
 * 
 */
package com.statnlp.example.linear_crf;

import java.util.ArrayList;
import java.util.List;

import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.NetworkIDMapper;
import com.statnlp.hybridnetworks.TableLookupNetwork;

/**
 * @author wei_lu
 *
 */
public class LinearCRFNetwork extends TableLookupNetwork{
	
	private static final long serialVersionUID = -269366520766930758L;
	public static boolean useZeroOneLossAtEachNode = true;
	
	private int _numNodes = -1;
	
	public LinearCRFNetwork(){
		
	}
	
	public LinearCRFNetwork(int networkId, LinearCRFInstance inst, LocalNetworkParam param){
		super(networkId, inst, param);
	}

	public LinearCRFNetwork(int networkId, LinearCRFInstance inst, long[] nodes, int[][][] children, LocalNetworkParam param, int numNodes){
		super(networkId, inst, nodes, children, param);
		this._numNodes = numNodes;
	}
	
	public double totalLossUpTo(int k, int[] child_k){
		if(useZeroOneLossAtEachNode){
			return zeroOneLossAtEachNode(k, child_k);
		} else {
			return zeroOneLossAtRootOnly(k, child_k);
		}
	}
	
	/**
	 * We implement zero-one loss here, with the loss applied at each node<br>
	 * So if the best path so far contains incorrect label, the loss is 1.0, otherwise 0.0.
	 * Another possibility is to calculate loss only at root node, effectively removing the loss
	 * from the Viterbi process during max-path finding, and only using the loss at the end<br>
	 * See {@link #zeroOneLossAtRootOnly(int, int[])}
	 * @param k
	 * @param child_k
	 * @return
	 */
	private double zeroOneLossAtEachNode(int k, int[] child_k){
		LinearCRFInstance inst = (LinearCRFInstance)this.getInstance();
		int size = inst.size();
		int[] nodeArr = getNodeArray(k);
		int pos = nodeArr[0]-1;
		Label predLabel = Label.get(nodeArr[1]);
		int nodeType = nodeArr[4];
		if(pos >= 0 && pos < size && nodeType != NODE_TYPES.ROOT.ordinal()){
			Label goldLabel = inst.output.get(pos);
			if(goldLabel == predLabel){
				// Same label, no change of loss
				return this._loss[child_k[0]];
			} else {
				return 1.0;
			}
		} else {
			if(child_k.length > 0){ // Root
				// At root, no change of loss
				return this._loss[child_k[0]];
			} else { // Leaf
				return 0.0;
			}
		}
	}
	
	/**
	 * We implement zero-one loss here, with the loss applied only at the root, which means we only
	 * evaluate the loss when the full structure is predicted, and not for partial structures<br>
	 * If the predicted structure contains incorrect label, the loss is 1.0, otherwise 0.0.
	 * Another possibility is to calculate loss at each node, effectively including the loss
	 * in the Viterbi process during max-path finding<br>
	 * See {@link #zeroOneLossAtEachNode(int, int[])}
	 * @param k
	 * @param child_k
	 * @return
	 */
	private double zeroOneLossAtRootOnly(int k, int[] child_k){
		LinearCRFInstance inst = (LinearCRFInstance)this.getInstance();
		int size = inst.size();
		int[] nodeArr = getNodeArray(k);
		int pos = nodeArr[0]-1;
		int nodeType = nodeArr[4];
		if(pos != size || nodeType != NODE_TYPES.ROOT.ordinal()){
			return 0.0;
		}
		List<Label> gold = inst.output;
		List<Label> pred = new ArrayList<Label>();
		int node_k = child_k[0];
		for(int i=size-1; i>=0; i--){
			int[] children_k = getMaxPath(node_k);
			if(children_k.length != 1){
				System.err.println("Child length not 1!");
			}
			node_k = children_k[0];
			long child = getNode(node_k);
			int[] child_arr = NetworkIDMapper.toHybridNodeArray(child);
			int childPos = child_arr[0]-1;
			if(childPos != i){
				System.err.println("Position encoded in the node array not the same as the interpretation!");
			}
			int tag_id = child_arr[1];
			pred.add(0, Label.get(tag_id));
		}
		
		if(gold.equals(pred)){
			return 0.0;
		} else {
			return 1.0;
		}
	}
	
	@Override
	public int countNodes(){
		if(this._numNodes==-1)
			return super.countNodes();
		return this._numNodes;
	}
	
	//remove the node k from the network.
//	@Override
	public void remove(int k){
		//DO NOTHING..
	}
	
	//check if the node k is removed from the network.
//	@Override
	public boolean isRemoved(int k){
		return false;
	}
	
}
