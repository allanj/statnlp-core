package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure.
 */
public class EdgeHypothesis extends Hypothesis{
	
	private double score;

	/**
	 * Creates an empty hypothesis with 0 score and null parent
	 */
	public EdgeHypothesis(int nodeIndex) {
		this(nodeIndex, null, 0.0);
	}
	
	/**
	 * Creates a hypothesis with a specific parent and specific score
	 * @param parent
	 * @param score
	 */
	public EdgeHypothesis(int nodeIndex, NodeHypothesis[] parents, double score){
		this.score = score;
		setNodeIndex(nodeIndex);
		setParents(parents);
		init();
	}
	
	/**
	 * @return the parent
	 */
	public NodeHypothesis[] parents() {
		return (NodeHypothesis[])parents;
	}
	
	public double score(){
		return this.score;
	}
	
	public IndexedScore getNextBestPath(){
		if(!hasMoreHypothesis){
			return null;
		} else if(lastBestIndex[0] == null){
			int[] bestIndex = new int[parents.length];
			Arrays.fill(bestIndex, 0);
			
			nextBestParentIndex.offer(IndexedScore.get(nodeIndex, bestIndex, this));
		} else {
			for(int i=0; i<lastBestIndex[0].index.length; i++){
				int[] newIndex = Arrays.copyOf(lastBestIndex[0].index, lastBestIndex[0].index.length);
				newIndex[i] += 1;
				IndexedScore nextBestCandidate = null;
				nextBestCandidate = IndexedScore.get(nodeIndex, newIndex, (EdgeHypothesis)this);
				if(nextBestCandidate != null && !presentInNextBestParentIndex.contains(nextBestCandidate)){
					presentInNextBestParentIndex.add(nextBestCandidate);
					nextBestParentIndex.offer(nextBestCandidate);
				}
			}
		}
		IndexedScore nextBestIndex = nextBestParentIndex.poll();
		if(nextBestIndex == null){
			hasMoreHypothesis = false;
			return null;
		}
		lastBestIndex[0] = nextBestIndex;
		presentInNextBestParentIndex.remove(nextBestIndex);
		bestParentList.add(nextBestIndex);
		return nextBestIndex;
	}
	
	public String toString(){
		int[] parentNodeIndex = new int[parents.length];
		for(int i=0; i<parents.length; i++){
			parentNodeIndex[i] = parents[i].nodeIndex();
		}
		return String.format("%d -> %s: %.3f", nodeIndex, Arrays.toString(parentNodeIndex), score);
	}

}
