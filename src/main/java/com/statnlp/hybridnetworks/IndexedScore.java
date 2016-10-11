/**
 * 
 */
package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * A wrapper for Hypothesis object with additional index specifying its
 * order in a top-k list
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
	
	public static IndexedScore get(int node_k, int[] index, EdgeHypothesis hypothesis){
		Hypothesis[] parents = hypothesis.parents;
		double score = hypothesis.score();
		for(int i=0; i<index.length; i++){
			IndexedScore kthBestParentAtIthPos = parents[i].getKthBestHypothesis(index[i]);
			if(kthBestParentAtIthPos == null){
				return null;
			}
			score += kthBestParentAtIthPos.score;
		}
		return new IndexedScore(node_k, score, index);
	}
	
	public static IndexedScore get(int node_k, int[] index, NodeHypothesis hypothesis){
		Hypothesis[] parents = hypothesis.parents;
		IndexedScore kthBestParentAtIthPos = parents[index[0]].getKthBestHypothesis(index[1]);
		if(kthBestParentAtIthPos == null){
			return null;
		}
		double score = kthBestParentAtIthPos.score;
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
