package com.statnlp.example.linear_ne;

import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkIDMapper;
import com.statnlp.neural.NeuralConfig;

public class ECRFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FEATYPE {local,entity, neural};
	private String OUT_SEP = NeuralConfig.OUT_SEP; 
	private String IN_SEP = NeuralConfig.IN_SEP;
	
	public ECRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	
	
	
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		
		//the tag in the sentence is actually the caps id.
		
		ECRFInstance inst = ((ECRFInstance)network.getInstance());
		//int instanceId = inst.getInstanceId();
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int pos = nodeArr[0]-1;
		int eId = nodeArr[1];
		if(pos<0 || pos >= inst.size() || eId==Entity.Entities.size())
			return FeatureArray.EMPTY;
			
		
		int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		int childEId = child[1];
//		int childPos = child[0]-1;
		
		String lw = pos>0? sent.get(pos-1).getName():"1738";
		String llw = pos==0? "1738": pos==1? "1738":sent.get(pos-2).getName();
		String llt = pos==0? "1": pos==1? "1":sent.get(pos-2).getTag();
		String lt = pos>0? sent.get(pos-1).getTag():"1";
		String rw = pos<sent.length()-1? sent.get(pos+1).getName():"1738";
		String rt = pos<sent.length()-1? sent.get(pos+1).getTag():"1";
		String rrw = pos==sent.length()-1? "1738": pos==sent.length()-2? "1738":sent.get(pos+2).getName();
		String rrt = pos==sent.length()-1? "1": pos==sent.length()-2? "1":sent.get(pos+2).getTag();
		
		String currWord = inst.getInput().get(pos).getName();
		String currTag = inst.getInput().get(pos).getTag();
//		String childWord = childPos>=0? inst.getInput().get(childPos).getName():"STR";
//		String childTag = childPos>=0? inst.getInput().get(childPos).getTag():"STR";
		
		
		
		
		String currEn = Entity.get(eId).getForm();
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
//			featureList.add(this._param_g.toFeature(network,FEATYPE.neural.name(), currEn,  currWord));
			featureList.add(this._param_g.toFeature(network, FEATYPE.neural.name(), currEn, llw+IN_SEP+lw+IN_SEP+currWord+IN_SEP+rw+IN_SEP+rrw+OUT_SEP+
										llt+IN_SEP+lt+IN_SEP+currTag+IN_SEP+rt+IN_SEP+rrt));
		}else{
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), currEn,  	currWord));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ET",	currEn+":"+currTag));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELW",	currEn+":"+lw));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELT",	currEn+":"+lt));
			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ERW",	currEn+":"+rw));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ERT",	currEn+":"+rt));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "ELT-T",	currEn+":"+lt+","+currTag));
			/****Add some prefix features******/
			for(int plen = 1;plen<=6;plen++){
				if(currWord.length()>=plen){
					String suff = currWord.substring(currWord.length()-plen, currWord.length());
					featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "E-PATTERN-SUFF-"+plen, currEn+":"+suff));
					String pref = currWord.substring(0,plen);
					featureList.add(this._param_g.toFeature(network,FEATYPE.local.name(), "E-PATTERN-PREF-"+plen, currEn+":"+pref));
				}
			}
			String prevEntity = Entity.get(childEId).getForm();

			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "E-prev-E",prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "currW-prevE-currE",currWord+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevW-prevE-currE",lw+":"+prevEntity+":"+currEn));
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "nextW-prevE-currE",rw+":"+prevEntity+":"+currEn));
			
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "currT-prevE-currE",currTag+":"+prevEntity+":"+currEn));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevT-prevE-currE",lt+":"+prevEntity+":"+currEn));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "nextT-prevE-currE",rt+":"+prevEntity+":"+currEn));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity.name(), "prevT-currT-prevE-currE",lt+":"+currTag+":"+prevEntity+":"+currEn));
		}
		
					

		
		
		
		
		
		ArrayList<Integer> finalList = new ArrayList<Integer>();
		for(int i=0;i<featureList.size();i++){
			if(featureList.get(i)!=-1)
				finalList.add(featureList.get(i));
		}
		int[] features = new int[finalList.size()];
		for(int i=0;i<finalList.size();i++) features[i] = finalList.get(i);
		if(features.length==0) return FeatureArray.EMPTY;
		fa = new FeatureArray(features);
		
		return fa;
	}
	

}
