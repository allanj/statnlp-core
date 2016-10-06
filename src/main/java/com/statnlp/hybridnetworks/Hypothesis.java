package com.statnlp.hybridnetworks;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure.
 */
public class Hypothesis implements Comparable<Hypothesis>{
	
	private double score;
	private Hypothesis parent;

	public Hypothesis() {
		score = 0.0;
	}
	
	public double addScore(double score){
		this.score += score;
		return this.score;
	}
	
	public double getScore(){
		return score;
	}

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(Hypothesis o) {
		return Double.compare(-score, -o.score);
	}

}
