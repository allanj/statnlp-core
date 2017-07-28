package com.statnlp.example.linear_ne;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.hypergraph.FeatureArray;
import com.statnlp.hypergraph.FeatureManager;
import com.statnlp.hypergraph.GlobalNetworkParam;
import com.statnlp.hypergraph.Network;
import com.statnlp.hypergraph.NetworkConfig;
import com.statnlp.hypergraph.NetworkIDMapper;
import com.statnlp.hypergraph.neural.MultiLayerPerceptron;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FeaType{
		word, tag, lw, lt, ltt, rw, rt, prefix, suffix,
		pairwise};
	private String OUT_SEP = MultiLayerPerceptron.OUT_SEP; 
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	private final String START = "STR";
	private final String END = "END";
	
	private String neuralType;
	private boolean moreBinaryFeatures = false;
	private int maxmimumPrefixSUffixLength = 6;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, String neuralType, boolean moreBinaryFeatures) {
		super(param_g);
		this.neuralType = neuralType;
		this.moreBinaryFeatures = moreBinaryFeatures;
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		ECRFInstance inst = ((ECRFInstance)network.getInstance());
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int pos = nodeArr[0]-1;
		int eId = nodeArr[1];
		if(pos < 0 || pos > inst.size() || ( pos != inst.size() && eId == Entity.Entities.size())) {
			return FeatureArray.EMPTY;
		}
		if (pos == inst.size() && eId != Entity.Entities.size()) return FeatureArray.EMPTY;
		String entity = pos != inst.size() ? Entity.get(eId).getForm() : END;
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
		
		String word = pos != inst.size() ? inst.getInput().get(pos).getForm() : END;
		String tag = pos != inst.size() ?  inst.getInput().get(pos).getTag() : END;
		String lw = pos>0? sent.get(pos-1).getForm() : START;
		String llw = pos==0? START: pos==1? START:sent.get(pos-2).getForm();
		String llt = pos==0? START: pos==1? START:sent.get(pos-2).getTag();
		String lt = pos>0? sent.get(pos-1).getTag():START;
		String rw = pos<sent.length()-1? sent.get(pos+1).getForm():END;
		String rt = pos<sent.length()-1? sent.get(pos+1).getTag():END;
		String rrw = pos==sent.length()-1? END: pos==sent.length()-2? END:sent.get(pos+2).getForm();
		String rrt = pos==sent.length()-1? END: pos==sent.length()-2? END:sent.get(pos+2).getTag();
		
		if (pos != inst.size()) {
			if(NetworkConfig.USE_NEURAL_FEATURES){
				Object input = null;
				if(neuralType.equals("lstm")) {
					input = new SimpleImmutableEntry<String, Integer>(sent.toString().toLowerCase(), pos);
				} else if(neuralType.equals("mlp")){
					input = llw+IN_SEP+lw+IN_SEP+word+IN_SEP+rw+IN_SEP+rrw+OUT_SEP+llt+IN_SEP+lt+IN_SEP+tag+IN_SEP+rt+IN_SEP+rrt;
				} else {
					input = word;
				}
				this.addNeural(network, 0, parent_k, children_k_index, input, eId);
			} else {
				featureList.add(this._param_g.toFeature(network,FeaType.word.name(), entity,  	word));
			}
		}
		String prevEntity = pos == 0 ? "STR" : Entity.get(childEId).getForm() + "";
		featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name(), entity,  prevEntity));
		
		if (pos != inst.size() && moreBinaryFeatures) {
			featureList.add(this._param_g.toFeature(network,FeaType.tag.name(), entity,	tag));
			featureList.add(this._param_g.toFeature(network,FeaType.lw.name(), 	entity,	lw));
			featureList.add(this._param_g.toFeature(network,FeaType.lt.name(), 	entity,	lt));
			featureList.add(this._param_g.toFeature(network,FeaType.rw.name(), 	entity,	rw));
			featureList.add(this._param_g.toFeature(network,FeaType.rt.name(), 	entity,	rt));
			featureList.add(this._param_g.toFeature(network,FeaType.ltt.name(), entity,	lt+","+tag));
			/****Add some prefix features******/
			for(int plen = 1;plen <= maxmimumPrefixSUffixLength; plen++){
				if(word.length()>=plen){
					String suff = word.substring(word.length()-plen, word.length());
					featureList.add(this._param_g.toFeature(network,FeaType.suffix.name()+plen, entity, suff));
					String pref = word.substring(0,plen);
					featureList.add(this._param_g.toFeature(network,FeaType.prefix.name()+plen, entity, pref));
				}
			}
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"-wle", entity, word+":"+prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"lwle", entity,	lw+":"+prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"rwre", entity,	rw+":"+prevEntity));
			
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"tle", 	entity,tag + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"ltle", entity,lt + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"rtle", entity,rt + ":" + prevEntity));
			featureList.add(this._param_g.toFeature(network,FeaType.pairwise.name()+"lttle",entity,lt + ":" + tag + ":" + prevEntity));
		}
		fa = this.createFeatureArray(network, featureList);
		return fa;
	}
}
