/**
 * 
 */
package com.statnlp.hybridnetworks;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * 
 */
public abstract class Hypothesis {

	/** The node index in which this hypothesis is applicable */
	protected int nodeIndex;
	/**
	 * The parents of this hypothesis, which is the previous partial hypothesis
	 */
	protected Hypothesis[] parents;
	protected IndexedScore[] lastBestIndex;
	protected boolean hasMoreHypothesis;
	protected PriorityQueue<IndexedScore> nextBestParentIndex;
	protected Set<IndexedScore> presentInNextBestParentIndex;
	protected ArrayList<IndexedScore> bestParentList;

	protected void init() {
		nextBestParentIndex = new PriorityQueue<IndexedScore>();
		lastBestIndex = new IndexedScore[1];
		bestParentList = new ArrayList<IndexedScore>();
		presentInNextBestParentIndex = new HashSet<IndexedScore>();
		hasMoreHypothesis = true;
	}
	
	public IndexedScore getKthBestHypothesis(int k){
		// Assuming the k is 0-based. So k=0 will return the best prediction
		while(bestParentList.size() <= k){
			IndexedScore nextBest = getNextBestPath();
			if(nextBest == null){
				return null;
			}
//			System.out.println("["+this+"] Generate the "+(k+1)+"-th best");
		}
		return bestParentList.get(k);
	}
	
	/**
	 * Return the next best path and add to the bestParentList, or return null
	 * @return
	 */
	public abstract IndexedScore getNextBestPath();
	
	public int nodeIndex(){
		return this.nodeIndex;
	}
	
	public void setNodeIndex(int nodeIndex){
		this.nodeIndex = nodeIndex;
	}
	
	/**
	 * @return the parent
	 */
	public Hypothesis[] parents() {
		return (NodeHypothesis[])parents;
	}
	
	/**
	 * @param parent the parent to set
	 */
	public void setParents(Hypothesis[] parents) {
		this.parents = parents;
	}
	
	public void setLastBestIndex(IndexedScore[] lastBestIndex){
		this.lastBestIndex = lastBestIndex;
	}

	public ArrayList<IndexedScore> bestParentList() {
		return bestParentList;
	}

	public void setBestParentList(ArrayList<IndexedScore> bestParentList) {
		this.bestParentList = bestParentList;
	}

	public PriorityQueue<IndexedScore> nextBestParentIndex() {
		return nextBestParentIndex;
	}

	public void setNextBestParentIndex(PriorityQueue<IndexedScore> nextBestParentIndex) {
		this.nextBestParentIndex = nextBestParentIndex;
	}

	public Set<IndexedScore> presentInNextBestParentIndex() {
		return presentInNextBestParentIndex;
	}

	public void setPresentInNextBestParentIndex(Set<IndexedScore> presentInNextBestParentIndex) {
		this.presentInNextBestParentIndex = presentInNextBestParentIndex;
	}

}
