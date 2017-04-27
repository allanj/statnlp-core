/**
 * 
 */
package com.statnlp.hybridnetworks;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure.
 * There are two types of Hypothesis: {@link NodeHypothesis} and {@link EdgeHypothesis}
 */
public abstract class Hypothesis {

	/**
	 * The node index in which this hypothesis is applicable
	 * For NodeHypothesis, this represents that node's index.
	 * For EdgeHypothesis, this represents the node index of the parent node.
	 */
	protected int nodeIndex;
	/**
	 * The children of this hypothesis, which is the previous partial hypothesis.
	 * For EdgeHypothesis, the children would be a list of NodeHypothesis.
	 * Similarly, for NodeHypothesis, the children would be a list of EdgeHypothesis.
	 */
	protected Hypothesis[] children;
	/**
	 * A variable to store the last best index.
	 * An array is used so that the reference to this member stays the same even though the 
	 * actual best index changes each time we get the next best index. 
	 */
	protected IndexedScore[] lastBestIndex;
	/**
	 * Whether there are more hypothesis to be predicted.
	 * Note that this is different from simply checking whether the queue is empty,
	 * because the queue is populated only when necessary by looking at the last best index.
	 */
	protected boolean hasMoreHypothesis;
	/**
	 * The priority queue storing the possible next best child.
	 * Since this is a priority queue, the next best child is the one in front of the queue.
	 */
	protected PriorityQueue<IndexedScore> nextBestChildQueue;
	/**
	 * A set to contain the list of candidates present in the priority queue.
	 * This is used to prevent the same candidate being entered into the queue
	 * multiple times from different neighbors.
	 */
	protected Set<IndexedScore> candidatesPresentInQueue;
	/**
	 * The cache to store the list of best children, which will contain the list of 
	 * best children up to the highest k on which {@link #getKthBestHypothesis(int)} has been called.
	 */
	protected ArrayList<IndexedScore> bestChildrenList;

	protected void init() {
		nextBestChildQueue = new PriorityQueue<IndexedScore>();
		lastBestIndex = new IndexedScore[1];
		bestChildrenList = new ArrayList<IndexedScore>();
		candidatesPresentInQueue = new HashSet<IndexedScore>();
		hasMoreHypothesis = true;
	}
	
	/**
	 * Returns the k-th best path at this hypothesis.
	 * @param k
	 * @return
	 */
	public IndexedScore getKthBestHypothesis(int k){
		// Assuming the k is 0-based. So k=0 will return the best prediction
		// Below we fill the cache until we satisfy the number of top-k paths requested.
		while(bestChildrenList.size() <= k){
			IndexedScore nextBest = getNextBestPath();
			if(nextBest == null){
				return null;
			}
			lastBestIndex[0] = nextBest;
			
			// Remove this candidate from the index to save memory.
			// Since this candidate has been selected as the best based on the scores in the priority queue,
			// that means all previous neighbors of this node has been selected, since they must have higher
			// scores compared to this. This means we no longer need to keep track of this candidate, since
			// this candidate will no longer be offered into the queue.
			// Remember that the purpose of this candidate index is to prevent the same candidate being entered 
			// into the priority queue multiple times.
			candidatesPresentInQueue.remove(nextBest);
			
			// Cache this next best candidate in the list
			bestChildrenList.add(nextBest);
//			System.out.println("["+this+"] Generated the "+(k+1)+"-th best");
		}
		return bestChildrenList.get(k);
	}
	
	/**
	 * Return the next best path, or return null if there is no next best path.
	 * @return
	 */
	public abstract IndexedScore getNextBestPath();
	
	public int nodeIndex(){
		return this.nodeIndex;
	}
	
	/**
	 * Sets node index of this hypothesis accordingly.
	 * For NodeHypothesis, this should be that node's index.
	 * For EdgeHypothesis, this should be the node index of the parent node.
	 * @param nodeIndex
	 */
	public void setNodeIndex(int nodeIndex){
		this.nodeIndex = nodeIndex;
	}
	
	/**
	 * @return The children of this hypothesis.
	 */
	public Hypothesis[] children() {
		return children;
	}
	
	/**
	 * @param children The children to set
	 */
	public void setChildren(Hypothesis[] children) {
		this.children = children;
	}
	
	public void setLastBestIndex(IndexedScore[] lastBestIndex){
		this.lastBestIndex = lastBestIndex;
	}

	public ArrayList<IndexedScore> bestChildrenList() {
		return bestChildrenList;
	}

	public void setBestChildrenList(ArrayList<IndexedScore> bestChildrenList) {
		this.bestChildrenList = bestChildrenList;
	}

	public PriorityQueue<IndexedScore> nextBestChildQueue() {
		return nextBestChildQueue;
	}

	public void setNextBestChildQueue(PriorityQueue<IndexedScore> nextBestChildQueue) {
		this.nextBestChildQueue = nextBestChildQueue;
	}

	public Set<IndexedScore> candidatesPresentInQueue() {
		return candidatesPresentInQueue;
	}

	public void setCandidatesPresentInQueue(Set<IndexedScore> candidatesPresentInQueue) {
		this.candidatesPresentInQueue = candidatesPresentInQueue;
	}

}
