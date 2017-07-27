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
import com.statnlp.neural.MultiLayerPerceptron;
import com.statnlp.neural.NeuralNetworkFeatureValueProvider;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FEATYPE {local,entity, neural};
	private String OUT_SEP = MultiLayerPerceptron.OUT_SEP; 
	private String IN_SEP = MultiLayerPerceptron.IN_SEP;
	
	private NeuralNetworkFeatureValueProvider net;
	private String neuralType;
	private boolean moreBinaryFeatures = false;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g, String neuralType, boolean moreBinaryFeatures) {
		super(param_g);
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			this.net = (NeuralNetworkFeatureValueProvider) param_g.getFeatureValueProviders().get(0);
		}
		this.neuralType = neuralType;
		this.moreBinaryFeatures = moreBinaryFeatures;
	}
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k, int children_k_index) {
		
		ECRFInstance inst = ((ECRFInstance)network.getInstance());
		//int instanceId = inst.getInstanceId();
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
			
			
//		System.err.println(Arrays.toString(nodeArr) + Entity.get(eId).toString());
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
//		int childPos = child[0]-1;
		
		String lw = null;
		String llw = null;
		String llt = null;
		String lt = null;
		String rw = null;
		String rt = null;
		String rrw = null;
		String rrt = null;
		String currEn = pos != inst.size() ? Entity.get(eId).getForm() : "END";
		String currWord = pos != inst.size() ? inst.getInput().get(pos).getForm() : "END";
		String currTag = pos != inst.size() ?  inst.getInput().get(pos).getTag() : "END";
		if (pos != inst.size()) {
			if(neuralType.equalsIgnoreCase("mlp")){
				lw = pos>0? sent.get(pos-1).getForm():"1738";
				llw = pos==0? "1738": pos==1? "1738":sent.get(pos-2).getForm();
				llt = pos==0? "0": pos==1? "0":sent.get(pos-2).getTag();
				lt = pos>0? sent.get(pos-1).getTag():"0";
				rw = pos<sent.length()-1? sent.get(pos+1).getForm():"1738";
				rt = pos<sent.length()-1? sent.get(pos+1).getTag():"0";
				rrw = pos==sent.length()-1? "1738": pos==sent.length()-2? "1738":sent.get(pos+2).getForm();
				rrt = pos==sent.length()-1? "0": pos==sent.length()-2? "0":sent.get(pos+2).getTag();
			}
			if(NetworkConfig.USE_NEURAL_FEATURES){
				Object input = null;
				if(neuralType.equals("lstm")) {
					input = new SimpleImmutableEntry<String, Integer>(sent.toString(), pos);
				} else if(neuralType.equals("mlp")){
					input = llw+IN_SEP+lw+IN_SEP+currWord+IN_SEP+rw+IN_SEP+rrw+OUT_SEP+llt+IN_SEP+lt+IN_SEP+currTag+IN_SEP+rt+IN_SEP+rrt;
				} else {
					input = currWord;
				}
				net.addHyperEdge(network, parent_k, children_k_index, input, eId);
			} else {
				featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), currEn,  	currWord));
			}

		}
		String prevEntity = pos == 0 ? "STR" : Entity.get(childEId).getForm() + "";
		featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), currEn,  prevEntity));
		if (currEn.startsWith("END") && prevEntity.startsWith("I-")) {
			System.out.println(network.getInstance().isLabeled());
		}
		if (pos != inst.size() && moreBinaryFeatures) {
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ET",		currEn+":"+currTag));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELW",	currEn+":"+lw));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELT",	currEn+":"+lt));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ERW",	currEn+":"+rw));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ERT",	currEn+":"+rt));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELT-T",	currEn+":"+lt+","+currTag));
			/****Add some prefix features******/
			for(int plen = 1;plen<=6;plen++){
				if(currWord.length()>=plen){
					String suff = currWord.substring(currWord.length()-plen, currWord.length());
					featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "E-PATTERN-SUFF-"+plen, currEn+":"+suff));
					String pref = currWord.substring(0,plen);
					featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "E-PATTERN-PREF-"+plen, currEn+":"+pref));
				}
			}
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "currW-prevE-currE",currWord+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevW-prevE-currE",lw+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "nextW-prevE-currE",rw+":"+prevEntity+":"+currEn));
			
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "currT-prevE-currE",currTag+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevT-prevE-currE",lt+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "nextT-prevE-currE",rt+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevT-currT-prevE-currE",lt+":"+currTag+":"+prevEntity+":"+currEn));
		}
		
		fa = this.createFeatureArray(network, featureList);
		return fa;
	}
}
