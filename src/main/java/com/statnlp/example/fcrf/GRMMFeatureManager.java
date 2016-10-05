package com.statnlp.example.fcrf;

import java.util.ArrayList;

import com.statnlp.commons.types.Sentence;
import com.statnlp.example.fcrf.TFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkIDMapper;

public class GRMMFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	public enum FEATYPE {grmm};
	
	public GRMMFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}
	//
	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		// TODO Auto-generated method stub
		TFInstance inst = ((TFInstance)network.getInstance());
		//int instanceId = inst.getInstanceId();
		Sentence sent = inst.getInput();
		long node = network.getNode(parent_k);
		int[] nodeArr = NetworkIDMapper.toHybridNodeArray(node);
		
		FeatureArray fa = null;
		ArrayList<Integer> featureList = new ArrayList<Integer>();
		
		int pos = nodeArr[0]-1;
		if(pos<0 || pos >= inst.size() || nodeArr[1]==NODE_TYPES.TAG_IN_E.ordinal() || nodeArr[1]==NODE_TYPES.E_IN_TAG.ordinal())
			return FeatureArray.EMPTY;
		
		int eId = nodeArr[2];
		//System.err.println(Arrays.toString(nodeArr));
		//int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		
		//int childPos = child[0]-1;
		
		if(nodeArr[1]==NODE_TYPES.ENODE_HYP.ordinal() ){
			String[] fs = sent.get(pos).getFS();
			for(String f: fs)
				featureList.add(this._param_g.toFeature(network, FEATYPE.grmm.name(), "GRMM", f+":"+Entity.get(eId).getForm()));
		}
		
		if(nodeArr[1]==NODE_TYPES.TNODE_HYP.ordinal()){
			String[] fs = sent.get(pos).getFS();
			for(String f: fs)
				featureList.add(this._param_g.toFeature(network, FEATYPE.grmm.name(), "GRMM", f+":"+Tag.get(eId).getForm()));
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
