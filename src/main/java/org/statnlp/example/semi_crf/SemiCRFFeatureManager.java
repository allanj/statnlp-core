package org.statnlp.example.semi_crf;

import java.util.ArrayList;

import org.statnlp.commons.types.Sentence;
import org.statnlp.example.base.BaseNetwork;
import org.statnlp.example.semi_crf.SemiCRFNetworkCompiler.NodeType;
import org.statnlp.hypergraph.FeatureArray;
import org.statnlp.hypergraph.FeatureBox;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.Network;

public class SemiCRFFeatureManager extends FeatureManager {
	
	private static final long serialVersionUID = 6510131496948610905L;
	private int prefixSuffixLen = 3;

	public enum FeaType {
		seg_prev_word,
		seg_prev_word_shape,
		seg_prev_tag,
		seg_next_word,
		seg_next_word_shape,
		seg_next_tag,
		segment,
		seg_len,
		start_word,
		start_tag,
		end_word,
		end_tag,
		word,
		tag,
		shape,
		seg_pref,
		seg_suff,
		transition,
		neural,
		head_word,
		head_tag,
		dep_word_label,
		dep_tag_label,
		cheat,
		modifier_word,
		modifier_tag
	}
	
	private final boolean CHEAT = false;
	
	public SemiCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
		if(CHEAT)
			System.out.println("[Info] Using the cheat features now..");
	}
	
	@Override
	protected FeatureArray extract_helper(Network net, int parent_k, int[] children_k, int children_k_index) {
		BaseNetwork network = (BaseNetwork)net;
		SemiCRFInstance instance = (SemiCRFInstance)network.getInstance();
		
		Sentence sent = instance.getInput();
		
		
		int[] parent_arr = network.getNodeArray(parent_k);
		int parentPos = parent_arr[0] - 1;
		
		NodeType parentType = NodeType.values()[parent_arr[2]];
		int parentLabelId = parent_arr[1];
		
		//since unigram, root is not needed
		if(parentType == NodeType.LEAF || parentType == NodeType.ROOT){
			return FeatureArray.EMPTY;
		}
		int[] child_arr = network.getNodeArray(children_k[0]);
		int childPos = child_arr[0] + 1 - 1;
		NodeType childType = NodeType.values()[child_arr[2]];
		int childLabelId = child_arr[1];

		if(CHEAT){
			//int instanceId = Math.abs(instance.getInstanceId());
			//int cheatFeature = _param_g.toFeature(network, FeatureType.cheat.name(), parentLabelId+"", instanceId+" "+parentPos);
			int cheatFeature = _param_g.toFeature(network, FeaType.cheat.name(), "cheat", "cheat");
			return new FeatureArray(new int[]{cheatFeature});
		}
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int start = childPos;
		int end = parentPos;
		if(parentPos==0 || childType==NodeType.LEAF ) start = childPos;
		String currEn = Label.get(parentLabelId).getForm();
		
		String lw = start>0? sent.get(start-1).getForm():"STR";
//		String lt = start>0? sent.get(start-1).getTag():"STR";
		String rw = end<sent.length()-1? sent.get(end+1).getForm():"END";
//		String rt = end<sent.length()-1? sent.get(end+1).getTag():"END";
		featureList.add(this._param_g.toFeature(network, FeaType.seg_prev_word.name(), 		currEn,	lw));
//		featureList.add(this._param_g.toFeature(network, FeaType.seg_prev_tag.name(), 		currEn, lt));
		featureList.add(this._param_g.toFeature(network, FeaType.seg_next_word.name(), 		currEn, rw));
//		featureList.add(this._param_g.toFeature(network, FeaType.seg_next_tag.name(), 	currEn, rt));
		
		StringBuilder segPhrase = new StringBuilder(sent.get(start).getForm());
		for(int pos=start+1;pos<=end; pos++){
			String w = sent.get(pos).getForm();
			segPhrase.append(" "+w);
		}
		featureList.add(this._param_g.toFeature(network, FeaType.segment.name(), currEn,	segPhrase.toString()));
		
		int lenOfSeg = end-start+1;
		featureList.add(this._param_g.toFeature(network, FeaType.seg_len.name(), currEn, lenOfSeg+""));
		
		/** Start and end features. **/
		String startWord = sent.get(start).getForm();
//		String startTag = sent.get(start).getTag();
		featureList.add(this._param_g.toFeature(network, FeaType.start_word.name(),	currEn,	startWord));
//		featureList.add(this._param_g.toFeature(network, FeaType.start_tag.name(),	currEn,	startTag));
		String endW = sent.get(end).getForm();
//		String endT = sent.get(end).getTag();
		featureList.add(this._param_g.toFeature(network, FeaType.end_word.name(),		currEn,	endW));
//		featureList.add(this._param_g.toFeature(network, FeaType.end_tag.name(),		currEn,	endT));
		
		int insideSegLen = lenOfSeg; //Math.min(twoDirInsideLen, lenOfSeg);
		for (int i = 0; i < insideSegLen; i++) {
			featureList.add(this._param_g.toFeature(network, FeaType.word.name()+":"+i,		currEn, sent.get(start+i).getForm()));
//			featureList.add(this._param_g.toFeature(network, FeaType.tag.name()+":"+i,		currEn, sent.get(start+i).getTag()));

			featureList.add(this._param_g.toFeature(network, FeaType.word.name()+":-"+i,	currEn,	sent.get(start+lenOfSeg-i-1).getForm()));
//			featureList.add(this._param_g.toFeature(network, FeaType.tag.name()+":-"+i,		currEn,	sent.get(start+lenOfSeg-i-1).getTag()));
		}
		/** needs to be modified maybe ***/
		for(int i=0; i<prefixSuffixLen; i++){
			String prefix = segPhrase.substring(0, Math.min(segPhrase.length(), i+1));
			String suffix = segPhrase.substring(Math.max(segPhrase.length()-i-1, 0));
			featureList.add(this._param_g.toFeature(network, FeaType.seg_pref.name()+"-"+i,	currEn,	prefix));
			featureList.add(this._param_g.toFeature(network, FeaType.seg_suff.name()+"-"+i,	currEn,	suffix));
		}
		String prevEntity = Label.get(childLabelId).getForm();
		featureList.add(this._param_g.toFeature(network,FeaType.transition.name(), prevEntity+"-"+currEn,	""));
		
		
		ArrayList<Integer> finalList = new ArrayList<Integer>();
		for(int i=0;i<featureList.size();i++){
			if(featureList.get(i)!=-1)
				finalList.add(featureList.get(i));
		}
		int[] features = new int[finalList.size()];
		for(int i=0;i<finalList.size();i++) features[i] = finalList.get(i);
		if(features.length==0) return FeatureArray.EMPTY;
		fa = new FeatureArray(FeatureBox.getFeatureBox(features, this.getParams_L()[network.getThreadId()]));
		
		return fa;
		
	}


}
