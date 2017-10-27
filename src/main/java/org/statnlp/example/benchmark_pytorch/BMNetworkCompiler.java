package org.statnlp.example.benchmark_pytorch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;
import org.statnlp.util.Pipeline;


public class BMNetworkCompiler extends NetworkCompiler {

	private static final long serialVersionUID = -3604477993417532194L;

	protected List<String> labels;
	protected Map<String, Integer> labelId;
	
	protected final boolean DEBUG = true;

	protected enum NodeType {
		leaf, tag, root
	};
	
	static {
		NetworkIDMapper.setCapacity(new int[] {3, 20, 5});
	}
	
	public BMNetworkCompiler(List<String> labels) {
		this.labels = labels;
		this.labelId = new HashMap<>(labels.size());
		for (int i = 0; i < this.labels.size(); i++) {
			labelId.put(this.labels.get(i), i);
		}
	}

	public BMNetworkCompiler(Pipeline<?> pipeline) {
		super(pipeline);
	}
	
	protected long toNode_root (int size) {
		return toNode(size - 1, labelId.get(BMConfig.END), NodeType.root);
	}
	
	protected long toNode_tag (int pos, int labelId) {
		return toNode(pos, labelId, NodeType.tag);
	}
	
	protected long toNode_Leaf () {
		return toNode(0, labelId.get(BMConfig.START), NodeType.leaf);
	}
	
	protected long toNode(int pos, int labelId, NodeType nodeType) {
		return NetworkIDMapper.toHybridNodeID(new int[]{nodeType.ordinal(), pos, labelId});
	}
	
	@Override
	public BaseNetwork compileLabeled(int networkId, Instance inst, LocalNetworkParam param) {
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		BMInstance tagInst = (BMInstance)inst;
		long leaf = toNode_Leaf();
		builder.addNode(leaf);
		List<String> output =  tagInst.getOutput();
		long[] children = new long[]{leaf};
		for (int i = 0; i < inst.size(); i++) {
			String label = output.get(i);
			long tagNode = this.toNode_tag(i, this.labelId.get(label));
			builder.addNode(tagNode);
			builder.addEdge(tagNode, children);
			children = new long[]{tagNode};
		}
		long root = this.toNode_root(inst.size());
		builder.addNode(root);
		builder.addEdge(root, children);
		BaseNetwork labeledNetwork = builder.build(networkId, inst, param, this);
		if (DEBUG) {
			BaseNetwork unlabeledNetwork = this.compileUnlabeled(networkId, inst, param);
			if (!unlabeledNetwork.contains(labeledNetwork)) {
				throw new RuntimeException("the labeled network is not contained");
			}
		}
		return labeledNetwork;
	}

	@Override
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param) {
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder();
		long leaf = toNode_Leaf();
		builder.addNode(leaf);
		long[] children = new long[]{leaf};
		for (int i = 0; i < inst.size(); i++) {
			long[] current = new long[this.labels.size()];
			for (int l = 0; l < this.labels.size(); l++) {
				long tagNode =  this.toNode_tag(i, l);
				builder.addNode(tagNode);
				for (long child : children) {
					builder.addEdge(tagNode, new long[]{child});
				}
				current[l] = tagNode;
			}
			children = current;
		}
		long root = this.toNode_root(inst.size());
		builder.addNode(root);
		for (long child : children) {
			builder.addEdge(root, new long[]{child});
		}
		BaseNetwork unlabeledNetwork = builder.build(networkId, inst, param, this);
		return unlabeledNetwork;
	}
	
	@Override
	public Instance decompile(Network network) {
		BaseNetwork unlabeledNetwork = (BaseNetwork)network;
		Instance inst = network.getInstance();
		int size = inst.size();
		long rootNode = this.toNode_root(size);
		int currIdx = Arrays.binarySearch(unlabeledNetwork.getAllNodes(), rootNode);
		List<String> prediction = new ArrayList<>(size);
 		for (int i = 0; i < size; i++) {
			int[] children = unlabeledNetwork.getMaxPath(currIdx);
			int child = children[0];
			int[] childArr = unlabeledNetwork.getNodeArray(child);
			prediction.add(0, this.labels.get(childArr[2]));
			currIdx = child;
		}
 		inst.setPrediction(prediction);
		return inst;
	}

}
