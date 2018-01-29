package org.statnlp.example.semi_crf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.statnlp.commons.types.Instance;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.base.BaseNetwork.NetworkBuilder;
import org.statnlp.hypergraph.LocalNetworkParam;
import org.statnlp.hypergraph.Network;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkIDMapper;

public class SemiCRFNetworkCompiler extends NetworkCompiler {
	
	private final static boolean DEBUG = false;
	
	private static final long serialVersionUID = 6585870230920484539L;
	public int maxSize = 128;
	public int maxSegmentLength = 8;
	public long[] allNodes;
	public int[][][] allChildren;
	
	public enum NodeType {
		LEAF,
		NORMAL,
		ROOT,
	}
	
	static {
		NetworkIDMapper.setCapacity(new int[]{10000, 20, 100});
	}

	public SemiCRFNetworkCompiler(int maxSize, int maxSegLength) {
		this.maxSize = Math.max(maxSize, this.maxSize);
		maxSegmentLength = Math.max(maxSegLength, maxSegmentLength);
		System.out.println(String.format("Max size: %s, Max segment length: %s", maxSize, maxSegLength));
		System.out.println(Label.LABELS.toString());
		buildUnlabeled(); 
	}

	public BaseNetwork compileLabeled(int networkId, Instance inst, LocalNetworkParam param){
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder(); 
		SemiCRFInstance instance = (SemiCRFInstance)inst;
		int size = instance.size();
		List<Span> output = instance.getOutput();
		Collections.sort(output);
		long leaf = toNode_leaf();
		builder.addNode(leaf);
		long prevNode = leaf;
		
		for(Span span: output){
			int labelId = span.label.id;
			long end = toNode(span.end, labelId);
			builder.addNode(end);
			builder.addEdge(end, new long[]{prevNode});
			prevNode = end;
			
		}
		long root = toNode_root(size);
		builder.addNode(root);
		builder.addEdge(root, new long[]{prevNode});
		BaseNetwork network = builder.build(networkId, instance, param, this);
		if(DEBUG){
//			System.out.println(network);
			BaseNetwork unlabeled = compileUnlabeled(networkId, instance, param);
//			System.out.println("for instance: "+instance.getInput().toString());
			if(!unlabeled.contains(network)){
				System.out.println("not contains");
				
			}
		}
		return network;
	}
	
	public BaseNetwork compileUnlabeled(int networkId, Instance inst, LocalNetworkParam param){
		SemiCRFInstance instance = (SemiCRFInstance)inst;
		int size = instance.size();
		long root = toNode_root(size);
		int root_k = Arrays.binarySearch(allNodes, root);
		int numNodes = root_k + 1;
		return NetworkBuilder.quickBuild(networkId, instance, this.allNodes, this.allChildren, numNodes, param, this);
	}
	
	//for O label, should only with span length 1.
	private synchronized void buildUnlabeled(){
		NetworkBuilder<BaseNetwork> builder = NetworkBuilder.builder(); 
		long leaf = toNode_leaf();
		builder.addNode(leaf);
		List<Long> currNodes = new ArrayList<Long>();
		for(int pos=0; pos<maxSize; pos++){
			for(int labelId=0; labelId<Label.LABELS.size(); labelId++){
				long node = toNode(pos, labelId);
				if(labelId!=Label.LABELS.get("O").id){
					for(int prevPos=pos-1; prevPos >= pos-maxSegmentLength && prevPos >= 0; prevPos--){
						for(int prevLabelId=0; prevLabelId<Label.LABELS.size(); prevLabelId++){
							long prevBeginNode = toNode(prevPos, prevLabelId);
							if(builder.contains(prevBeginNode)){
								builder.addNode(node);
								builder.addEdge(node, new long[]{prevBeginNode});
							}
						}
					}
					if(pos>=0){
						builder.addNode(node);
						builder.addEdge(node, new long[]{toNode_leaf()});
					}
				}else{
					//O label should be with only length 1. actually does not really affect.
					int prevPos = pos - 1;
					if(prevPos>=0){
						for(int prevLabelId=0; prevLabelId<Label.LABELS.size(); prevLabelId++){
							long prevBeginNode = toNode(prevPos, prevLabelId);
							if(builder.contains(prevBeginNode)){
								builder.addNode(node);
								builder.addEdge(node, new long[]{prevBeginNode});
							}
						}
					}
					if(pos==0){
						builder.addNode(node);
						builder.addEdge(node, new long[]{toNode_leaf()});
					}
				}
				currNodes.add(node);
			}
			long root = toNode_root(pos+1);
			builder.addNode(root);
			for(long currNode: currNodes){
				if(builder.contains(currNode)){
					builder.addEdge(root, new long[]{currNode});
				}	
			}
			currNodes = new ArrayList<Long>();
		}
		BaseNetwork network = builder.buildRudimentaryNetwork();
		//sViewer.visualizeNetwork(network, null, "UnLabeled Network");
		allNodes = network.getAllNodes();
		allChildren = network.getAllChildren();
	}
	
	private long toNode_leaf(){
		return toNode(0, Label.get("O").id, NodeType.LEAF);
	}
	
	private long toNode(int pos, int labelId){
		return toNode(pos+1, labelId, NodeType.NORMAL);
	}
	
	private long toNode_root(int size){
		return toNode(size, Label.LABELS.size(), NodeType.ROOT);
	}
	
	private long toNode(int pos, int labelId, NodeType type){
		return NetworkIDMapper.toHybridNodeID(new int[]{pos, labelId, type.ordinal()});
	}

	@Override
	public SemiCRFInstance decompile(Network net) {
		BaseNetwork network = (BaseNetwork)net;
		SemiCRFInstance result = (SemiCRFInstance)network.getInstance().duplicate();
		List<Span> prediction = new ArrayList<Span>();
		int node_k = network.countNodes()-1;
		while(node_k > 0){
			int[] children_k = network.getMaxPath(node_k);
			int[] child_arr = network.getNodeArray(children_k[0]);
			int pos = child_arr[0] - 1;;
			
			int nodeType = child_arr[2];
			if(nodeType == NodeType.LEAF.ordinal()){
				break;
			} 
			int labelId = child_arr[1];
			//System.err.println(pos+","+Label.LABELS_INDEX.get(labelId).getForm()+" ," + nodeType.toString());
			int end = pos;
			if(end!=0){
				int[] children_k1 = network.getMaxPath(children_k[0]);
				int[] child_arr1 = network.getNodeArray(children_k1[0]);
				int start = child_arr1[0] + 1 - 1;
				if(child_arr1[2]==NodeType.LEAF.ordinal())
					start = child_arr1[0];
				prediction.add(new Span(start, end, Label.LABELS_INDEX.get(labelId)));
			}else{
				prediction.add(new Span(end, end, Label.LABELS_INDEX.get(labelId)));
			}
			node_k = children_k[0];
			
		}
		Collections.sort(prediction);
		result.setPrediction(prediction);
		return result;
	}

	public double costAt(Network network, int parent_k, int[] child_k){
		return 0.0;
	}
	
	
}
