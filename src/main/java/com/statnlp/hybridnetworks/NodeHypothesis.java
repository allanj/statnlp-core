package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure.
 */
public class NodeHypothesis extends Hypothesis{
	
	/**
	 * Creates an empty hypothesis with 0 score and null parent
	 */
	public NodeHypothesis(int nodeIndex) {
		this(nodeIndex, null);
	}
	
	/**
	 * Creates a hypothesis with a specific parent and specific score
	 * @param parent
	 * @param score
	 */
	public NodeHypothesis(int nodeIndex, EdgeHypothesis[] parents){
		setNodeIndex(nodeIndex);
		setParents(parents);
		init();
	}
	
	/**
	 * @return the parent
	 */
	public EdgeHypothesis[] parents() {
		return (EdgeHypothesis[])parents;
	}
	
	public IndexedScore getNextBestPath(){
		if(!hasMoreHypothesis){
			return null;
		} else if(lastBestIndex[0] == null){
			for(int i=0; i<parents.length; i++){
				nextBestParentIndex.offer(IndexedScore.get(nodeIndex, new int[]{i, 0}, this));
			}
		} else {
			int[] newIndex = Arrays.copyOf(lastBestIndex[0].index, lastBestIndex[0].index.length);
			newIndex[1] += 1;
			IndexedScore nextBestCandidate = IndexedScore.get(lastBestIndex[0].node_k, newIndex, (NodeHypothesis)this);
			if(nextBestCandidate != null && !presentInNextBestParentIndex.contains(nextBestCandidate)){
				presentInNextBestParentIndex.add(nextBestCandidate);
				nextBestParentIndex.offer(nextBestCandidate);
			}
		}
		IndexedScore nextBestIndex = nextBestParentIndex.poll();
		if(nextBestIndex == null){
			hasMoreHypothesis = false;
			return null;
		}
		presentInNextBestParentIndex.remove(nextBestIndex);
		lastBestIndex[0] = nextBestIndex;
		bestParentList.add(nextBestIndex);
		return nextBestIndex;
	}
	
	public String toString(){
		return String.format("%d -> %s", nodeIndex, Arrays.toString(parents));
	}

}
