/**
 * 
 */
package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * A wrapper for configuration object (the {@link #index} array) together with its score.
 * This is the object being put in the priority queue during top-k decoding.<br> 
 * This behaves differently when used in NodeHypothesis or EdgeHypothesis.
 */
public class IndexedScore implements Comparable<IndexedScore>{
	
	public double score;
	public int[] index;
	public int node_k;

	public IndexedScore(int node_k, double score, int[] index) {
		this.score = score;
		this.index = index;
		this.node_k = node_k;
	}
	
	/**
	 * Returns the hypothesis according to the index configuration.
	 * This will return the IndexedScore object corresponding to taking the k-th best
	 * path for each child node of the specified edge hypothesis.
	 * Note that the index array will be an array containing the k-th best path requested
	 * of each child node. 
	 * @param node_k The node index of the parent node of this edge.
	 * @param index The index configuration. The length should be the same as the number of children that
	 * 				this edge has. Each representing the k-th best path that should be considered.
	 * @param hypothesis The EdgeHypothesis object representing the specified edge.
	 * @return The path based on the configuration in index array. If the configuration is invalid,
	 * 		   this will return null.
	 */
	public static IndexedScore get(int node_k, int[] index, EdgeHypothesis hypothesis){
		Hypothesis[] children = hypothesis.children;
		double score = hypothesis.score();
		for(int i=0; i<index.length; i++){
			IndexedScore kthBestChildrenAtIthPos = children[i].getKthBestHypothesis(index[i]);
			if(kthBestChildrenAtIthPos == null){
				// If the request contains an invalid k-th best path for any child,
				// then return null, to say that this configuration is invalid (there is no k-th best
				// for the specified child).
				return null;
			}
			score += kthBestChildrenAtIthPos.score;
		}
		return new IndexedScore(node_k, score, index);
	}
	
	/**
	 * Returns the hypothesis according to the index configuration.
	 * This will return the k-th best path of the m-th edge of the specified node hypothesis,
	 * where k = index[1] and m = index[0].
	 * @param node_k The specified node index
	 * @param index Specifies the configuration to take. This should be a two-element array,
	 * 				where index[0] represents the m-th edge of this node,
	 * 				and index[1] represents the k-th best path of that edge.
	 * @param hypothesis The NodeHypothesis object representing the specified node.
	 * @return The path based on the configuration in index array. If the configuration is invalid,
	 * 		   this will return null.
	 */
	public static IndexedScore get(int node_k, int[] index, NodeHypothesis hypothesis){
		Hypothesis[] children = hypothesis.children;
		IndexedScore kthBestChildrenAtIthPos = children[index[0]].getKthBestHypothesis(index[1]);
		if(kthBestChildrenAtIthPos == null){
			return null;
		}
		double score = kthBestChildrenAtIthPos.score;
//		System.out.println(String.format("Best of %s: %.3f", hypothesis, score));
		return new IndexedScore(node_k, score, index);
	}
	
	public boolean equals(Object o){
		if(o instanceof IndexedScore){
			IndexedScore s = (IndexedScore)o;
			if(!Arrays.equals(index, s.index)){
				return false;
			}
			return s.node_k == node_k;
		}
		return false;
	}

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(IndexedScore o) {
		return Double.compare(o.score, this.score);
	}
	
	@Override
	public String toString(){
		return String.format("%.3f %s", score, Arrays.toString(index));
	}
	
	@Override
	public int hashCode(){
		return index.hashCode();
	}

}
