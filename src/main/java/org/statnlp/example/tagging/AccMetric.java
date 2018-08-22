package org.statnlp.example.tagging;

import org.statnlp.hypergraph.decoding.Metric;

public class AccMetric implements Metric {

	public double acc; 
	
	public AccMetric(double acc) {
		this.acc = acc;
	}

	@Override
	public boolean isBetter(Metric other) {
		if (this.acc > ((AccMetric)other).acc)
			return true;
		else return false;
	}

	@Override
	public Object getMetricValue() {
		return acc;
	}

}
