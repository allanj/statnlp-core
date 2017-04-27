package com.statnlp.hybridnetworks;

import java.util.Arrays;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure at a specific edge.
 */
public class EdgeHypothesis extends Hypothesis{
	
	private double score;

	/**
	 * Creates an empty hypothesis with 0 score and no children
	 */
	public EdgeHypothesis(int nodeIndex) {
		this(nodeIndex, null, 0.0);
	}
	
	/**
	 * Creates a hypothesis with specific children (the list of nodes connected to this hyperedge,
	 * excluding the parent) and specific score.
	 * @param parent
	 * @param score
	 */
	public EdgeHypothesis(int nodeIndex, NodeHypothesis[] children, double score){
		this.score = score;
		setNodeIndex(nodeIndex);
		setChildren(children);
		init();
	}
	
	public NodeHypothesis[] children() {
		return (NodeHypothesis[])children;
	}
	
	/**
	 * The score of this hypothesis, which is the sum of the scores from this point to the leaf node.
	 * @return
	 */
	public double score(){
		return this.score;
	}
	
	public IndexedScore getNextBestPath(){
		if(!hasMoreHypothesis){
			return null;
		} else if(lastBestIndex[0] == null){
			// This case means this is the first time this method is called, means
			// we are looking for the best path, so we take all the best from the child nodes
			// by setting the bestIndex array to all 0.
			int[] bestIndex = new int[children.length];
			Arrays.fill(bestIndex, 0);
			
			nextBestChildQueue.offer(IndexedScore.get(nodeIndex, bestIndex, this));
		} else {
			// The index length in this EdgeHypothesis represents the degree of this hyperedge minus one.
			// So this represents the number of child nodes connected to the single parent node.
			
			// Below, we consider the next best candidate for each child node in this hyperedge 
			// and put them to the candidate priority queue.
			for(int i=0; i<lastBestIndex[0].index.length; i++){
				int[] newIndex = Arrays.copyOf(lastBestIndex[0].index, lastBestIndex[0].index.length);
				newIndex[i] += 1;
				IndexedScore nextBestChildCandidate = IndexedScore.get(nodeIndex, newIndex, (EdgeHypothesis)this);
				if(nextBestChildCandidate != null && !candidatesPresentInQueue.contains(nextBestChildCandidate)){
					candidatesPresentInQueue.add(nextBestChildCandidate);
					nextBestChildQueue.offer(nextBestChildCandidate);
				}
			}
		}
		// After lazily expand the last best node, take the best scoring one.
		IndexedScore nextBestIndex = nextBestChildQueue.poll();
		if(nextBestIndex == null){
			hasMoreHypothesis = false;
			return null;
		}
		return nextBestIndex;
	}
	
	public String toString(){
		int[] parentNodeIndex = new int[children.length];
		for(int i=0; i<children.length; i++){
			parentNodeIndex[i] = children[i].nodeIndex();
		}
		return String.format("%d -> %s: %.3f", nodeIndex, Arrays.toString(parentNodeIndex), score);
	}

}
