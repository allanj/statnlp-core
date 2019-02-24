package org.statnlp.example.tagging;

import org.statnlp.hypergraph.decoding.Metric;

public class TagMetric implements Metric {

	
	double acc;
	public TagMetric(double acc) {
		this.acc = acc;
	}

	@Override
	public boolean isBetter(Metric other) {
		TagMetric tm = (TagMetric)other;
		return this.acc > tm.acc;
	}

	@Override
	public Object getMetricValue() {
		return this.acc;
	}

}
