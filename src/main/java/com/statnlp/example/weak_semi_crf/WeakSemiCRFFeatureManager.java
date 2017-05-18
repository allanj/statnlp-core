package com.statnlp.example.weak_semi_crf;

import java.util.ArrayList;

import com.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler.NodeType;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.util.Pipeline;

public class WeakSemiCRFFeatureManager extends FeatureManager {
	
	private static final long serialVersionUID = 6510131496948610905L;
	
	private static final boolean CHEAT = false;

	public enum FeatureType{
		CHEAT,
		
		SEGMENT,
		START_CHAR,
		END_CHAR,
		
		UNIGRAM,
		SUBSTRING,

		ENDS_WITH_SPACE,
		NUM_SPACES,
		
		PREV_WORD,
		START_BOUNDARY_WORD,
		WORDS,
		END_BOUNDARY_WORD,
		NEXT_WORD,
		
		BIGRAM,
	}
	
	public int unigramWindowSize = 5;
	public int substringWindowSize = 5;

	public WeakSemiCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	public WeakSemiCRFFeatureManager(Pipeline pipeline){
		this(pipeline.param);
	}
	
	@Override
	protected FeatureArray extract_helper(Network net, int parent_k, int[] children_k) {
		WeakSemiCRFNetwork network = (WeakSemiCRFNetwork)net;
		WeakSemiCRFInstance instance = (WeakSemiCRFInstance)network.getInstance();
		
		int[] parent_arr = network.getNodeArray(parent_k);
		int parentPos = parent_arr[0];
		NodeType parentType = NodeType.values()[parent_arr[1]];
		int parentLabelId = parent_arr[2];
		
		if(parentType == NodeType.LEAF || children_k.length == 0){
			return FeatureArray.EMPTY;
		}
		
		int[] child_arr = network.getNodeArray(children_k[0]);
		int childPos = child_arr[0];
		NodeType childType = NodeType.values()[child_arr[1]];
		int childLabelId = child_arr[2];

		GlobalNetworkParam param_g = this._param_g;
		int bigramFeature = param_g.toFeature(network, FeatureType.BIGRAM.name(), parentLabelId+"", parentLabelId+" "+childLabelId);
		if(parentType == NodeType.ROOT || childType == NodeType.LEAF){
			return new FeatureArray(new int[]{
					bigramFeature,
			});
		}
		
		if(CHEAT){
			int instanceId = Math.abs(instance.getInstanceId());
			int cheatFeature = param_g.toFeature(network, FeatureType.CHEAT.name(), parentLabelId+"", instanceId+" "+parentPos+" "+childPos+" "+parentLabelId+" "+childLabelId);
			return new FeatureArray(new int[]{cheatFeature});
		}
		
		String input = instance.input;
		String[] inputArr = instance.getInputAsArray();
		int length = input.length();
		int isSpaceFeature = param_g.toFeature(network, FeatureType.ENDS_WITH_SPACE.name(), parentLabelId+"", (inputArr[parentPos].equals(" "))+"");
		int startCharFeature = param_g.toFeature(network, FeatureType.START_CHAR.name(), parentLabelId+"", inputArr[childPos]);
		int endCharFeature = param_g.toFeature(network, FeatureType.END_CHAR.name(), parentLabelId+"", inputArr[parentPos]);
		int segmentFeature = param_g.toFeature(network, FeatureType.SEGMENT.name(), parentLabelId+"", input.substring(childPos, parentPos));
		
		String[] words = input.split(" ");
		int numSpaces = words.length-1;
		int numSpacesFeature = param_g.toFeature(network, FeatureType.NUM_SPACES.name(), parentLabelId+"", numSpaces+"");
		
		int prevSpaceIdx = input.lastIndexOf(' ', childPos-1);
		if(prevSpaceIdx == -1){
			prevSpaceIdx = 0;
		}
		int firstSpaceIdx = input.indexOf(' ', childPos);
		if(firstSpaceIdx == -1){
			firstSpaceIdx = prevSpaceIdx;
		}
		int prevWordFeature = param_g.toFeature(network, FeatureType.PREV_WORD.name(), parentLabelId+"", input.substring(prevSpaceIdx, childPos));
		int startBoundaryWordFeature = param_g.toFeature(network, FeatureType.START_BOUNDARY_WORD.name(), parentLabelId+"", input.substring(prevSpaceIdx, firstSpaceIdx));
		
		int nextSpaceIdx = input.indexOf(' ', parentPos+1);
		if(nextSpaceIdx == -1){
			nextSpaceIdx = length;
		}
		int lastSpaceIdx = input.lastIndexOf(' ', parentPos);
		if(lastSpaceIdx == -1){
			lastSpaceIdx = nextSpaceIdx;
		}
		int nextWordFeature = param_g.toFeature(network, FeatureType.NEXT_WORD.name(), parentLabelId+"", input.substring(parentPos+1, nextSpaceIdx));
		int endBoundaryWordFeature = param_g.toFeature(network, FeatureType.END_BOUNDARY_WORD.name(), parentLabelId+"", input.substring(lastSpaceIdx, nextSpaceIdx));
		
		ArrayList<Integer> features = new ArrayList<Integer>();
		features.add(bigramFeature);
		features.add(isSpaceFeature);
		features.add(startCharFeature);
		features.add(endCharFeature);
		features.add(segmentFeature);
		features.add(numSpacesFeature);
		features.add(prevWordFeature);
		features.add(nextWordFeature);
		features.add(startBoundaryWordFeature);
		features.add(endBoundaryWordFeature);
		
		int[] wordFeatures = new int[2*words.length];
		for(int i=0; i<words.length; i++){
			wordFeatures[i] = param_g.toFeature(network, FeatureType.WORDS.name()+i, parentLabelId+"", words[i]);
			wordFeatures[2*words.length-i-1] = param_g.toFeature(network, FeatureType.WORDS.name()+"-"+i, parentLabelId+"", words[i]);
		}
		for(int feature: wordFeatures){
			features.add(feature);
		}
		
		int unigramFeatureSize = 2*unigramWindowSize;
		int[] unigramFeatures = new int[unigramFeatureSize];
		for(int i=0; i<unigramWindowSize; i++){
			String curInput = "";
			if(parentPos+i+1 < length){
				curInput = inputArr[parentPos+i+1];
			}
			unigramFeatures[i] = param_g.toFeature(network, FeatureType.UNIGRAM+":"+i, parentLabelId+"", curInput);
			curInput = "";
			if(childPos-i-1 >= 0){
				curInput = inputArr[childPos-i-1];
			}
			unigramFeatures[unigramFeatureSize-i-1] = param_g.toFeature(network, FeatureType.UNIGRAM+":-"+i, parentLabelId+"", curInput);
		}
		for(int feature: unigramFeatures){
			features.add(feature);
		}
		
		int substringFeatureSize = 2*substringWindowSize;
		int[] substringFeatures = new int[substringFeatureSize];
		for(int i=0; i<substringWindowSize; i++){
			String curInput = "";
			if(parentPos+i+1< length){
				curInput = input.substring(parentPos, parentPos+i+1);
			}
			substringFeatures[i] = param_g.toFeature(network, FeatureType.SUBSTRING+":"+i, parentLabelId+"", curInput);
			curInput = "";
			if(childPos-i-1 >= 0){
				curInput = input.substring(childPos-i-1, childPos);
			}
			substringFeatures[unigramFeatureSize-i-1] = param_g.toFeature(network, FeatureType.SUBSTRING+":-"+i, parentLabelId+"", curInput);
		}
		for(int feature: substringFeatures){
			features.add(feature);
		}
		
		int[] featureArr = new int[features.size()];
		int i=0;
		for(int feature: features){
			featureArr[i] = feature;
			i++;
		}
		return new FeatureArray(featureArr);
	}

}
