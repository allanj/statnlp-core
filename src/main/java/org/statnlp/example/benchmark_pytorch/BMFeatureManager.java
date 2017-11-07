package org.statnlp.example.benchmark_pytorch;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.benchmark_pytorch.BMNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.util.instance_parser.InstanceParser;

public class BMFeatureManager extends FeatureManager {

	private static final long serialVersionUID = -6059629463406022487L;

	private List<String> labels;
	
	public BMFeatureManager(GlobalNetworkParam param_g, List<String> labels) {
		super(param_g);
		this.labels = labels;
	}

	public BMFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser) {
		super(param_g, instanceParser);
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		int[] paArr = network.getNodeArray(parent_k);
		NodeType parentNodeType = NodeType.values()[paArr[0]];
		if (parentNodeType == NodeType.leaf)
			return FeatureArray.EMPTY;
		List<Integer> fs = new ArrayList<>();
		BMInstance inst = (BMInstance) network.getInstance();
		Sentence sent = inst.getInput();
		
		int pos = paArr[1];
		int labelId = paArr[2];
		String output = parentNodeType == NodeType.root ? BMConfig.END : this.labels.get(labelId);
		
		if (NetworkConfig.USE_NEURAL_FEATURES && parentNodeType != NodeType.root) {
			//BiLSTM: position, sentence
			SimpleImmutableEntry<String, Integer> edgeInput = 
					new SimpleImmutableEntry<String, Integer>(sent.toString(), pos);
			this.addNeural(network, 0, parent_k, children_k_index, edgeInput, labelId);
		}
		int[] childArr = network.getNodeArray(children_k[0]);
		int childLabelId = childArr[2];
		String childLabel = this.labels.get(childLabelId);
		fs.add(this._param_g.toFeature(network, "transition", output, childLabel));
		return this.createFeatureArray(network, fs);
	}

}
