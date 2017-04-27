package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure at a specific node.
 */
public class NodeHypothesis extends Hypothesis{
	
	/**
	 * Creates an empty hypothesis for the specified node index.
	 */
	public NodeHypothesis(int nodeIndex) {
		this(nodeIndex, null);
	}
	
	/**
	 * Creates a hypothesis with a specific children and specific node index
	 * @param parent
	 * @param score
	 */
	public NodeHypothesis(int nodeIndex, EdgeHypothesis[] children){
		setNodeIndex(nodeIndex);
		setChildren(children);
		init();
	}
	
	public EdgeHypothesis[] children() {
		return (EdgeHypothesis[])children;
	}
	
	public IndexedScore getNextBestPath(){
		if(!hasMoreHypothesis){
			return null;
		} else if(lastBestIndex[0] == null){
			// This case means this is the first time this method is called, means
			// we are looking for the best path, so we compare the best path from all
			// child hyperedges, and later we will take the best one.
			for(int i=0; i<children.length; i++){
				nextBestChildQueue.offer(IndexedScore.get(nodeIndex, new int[]{i, 0}, this));
			}
		} else {
			int[] newIndex = Arrays.copyOf(lastBestIndex[0].index, lastBestIndex[0].index.length);
			// Remember that the IndexedScore in NodeHypothesis contains an index with only two elements.
			// The first element represent the edge id, and the second element represent the k-th best candidate from that edge
			
			// Below, since we have "consumed" current edge, then we consider the next best one of that edge.
			newIndex[1] += 1;
			IndexedScore nextBestCandidate = IndexedScore.get(lastBestIndex[0].node_k, newIndex, (NodeHypothesis)this);
			if(nextBestCandidate != null && !candidatesPresentInQueue.contains(nextBestCandidate)){
				candidatesPresentInQueue.add(nextBestCandidate);
				nextBestChildQueue.offer(nextBestCandidate);
			}
		}
		IndexedScore nextBestIndex = nextBestChildQueue.poll();
		if(nextBestIndex == null){
			hasMoreHypothesis = false;
			return null;
		}
		return nextBestIndex;
	}
	
	public String toString(){
		return String.format("%d -> %s", nodeIndex, Arrays.toString(children));
	}

}
