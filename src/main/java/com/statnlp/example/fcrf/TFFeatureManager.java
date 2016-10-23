package com.statnlp.example.fcrf;

import java.util.ArrayList;
import java.util.Arrays;

import com.statnlp.commons.types.Sentence;
import com.statnlp.example.fcrf.TFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkIDMapper;
import com.statnlp.neural.NeuralConfig;

public class TFFeatureManager extends FeatureManager {

	private static final long serialVersionUID = 376931974939202432L;

	

	private String IN_SEP = NeuralConfig.IN_SEP;
	public enum FEATYPE {
		entity_currWord,
		entity_leftWord1,
		entity_leftWord2,
		entity_leftWord3,
		entity_rightWord1,
		entity_rightWord2,
		entity_rightWord3,
		entity_type1,
		entity_type2,
		entity_type3,
		entity_type4,
		entity_type5,
		tag_currWord,
		tag_leftWord1,
		tag_leftWord2,
		tag_leftWord3,
		tag_rightWord1,
		tag_rightWord2,
		tag_rightWord3,
		tag_type1,
		tag_type2,
		tag_type3,
		tag_type4,
		tag_type5,
		joint,
		neural_1,
		neural_2};
	
		
//	public PrintWriter pw;
	public TFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
//		try {
//			pw = RAWF.writer("data/dat/grmmtrain.txt");
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
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
		if(pos<0 || pos >= inst.size())
			return FeatureArray.EMPTY;
		
		int eId = nodeArr[2];
		//System.err.println(Arrays.toString(nodeArr));
		//int[] child = NetworkIDMapper.toHybridNodeArray(network.getNode(children_k[0]));
		
		//int childPos = child[0]-1;
		
		if(nodeArr[1]==NODE_TYPES.ENODE.ordinal()){
			//must be in the ne chain and not tag in e node
			addEntityFeatures(featureList, network, sent, pos, eId);
			//false: means it's NE structure
			addJointFeatures(featureList, network, sent, pos, eId, parent_k, children_k, false);
			
		}
		
		if(nodeArr[1]==NODE_TYPES.TNODE.ordinal()){
			addPOSFeatures(featureList, network, sent, pos, eId);
			addJointFeatures(featureList, network, sent, pos, eId, parent_k, children_k, true);
		}
		
		ArrayList<Integer> finalList = new ArrayList<Integer>();
		for(int i=0;i<featureList.size();i++){
			if(featureList.get(i)!=-1)
				finalList.add(featureList.get(i));
		}
		int[] features = new int[finalList.size()];
		for(int i=0;i<finalList.size();i++) features[i] = finalList.get(i);
		
		
		//if(features.length==0) return FeatureArray.EMPTY;
		fa = new FeatureArray(features);
		//printing the features.
		
		/****
		// Printing the features: for comparsion with GRMM only
		 
		if(network.getInstance().isLabeled() && network.getInstance().getInstanceId()>0){
			if(nodeArr[1]==NODE_TYPES.ENODE_HYP.ordinal()){
				pw.write(sent.get(pos).getTag()+" "+sent.get(pos).getEntity()+" ----");
				int[] fs = fa.getCurrent();
				for(int f : fs)
					pw.write(" "+f);
			}
			if(nodeArr[1]==NODE_TYPES.TNODE_HYP.ordinal()){
				int[] fs = fa.getCurrent();
				for(int f : fs)
					pw.write(" "+f);
				pw.write("\n");
				if(pos==sent.length()-1)
					pw.write("\n");
			}
		}
		***/
		return fa;
	}
	
	private void addEntityFeatures(ArrayList<Integer> featureList,Network network, Sentence sent, int pos, int eId){
		
		if(eId!=(Entity.ENTS.size()+Tag.TAGS.size())){
			String lw = pos>0? sent.get(pos-1).getName():"STR";
			String rw = pos<sent.length()-1? sent.get(pos+1).getName():"END";
			String currWord = sent.get(pos).getName();
			String currEn = Entity.get(eId).getForm();
			
			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_currWord.name(), 	currEn,	currWord));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_leftWord1.name(), 	currEn,	lw));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_leftWord2.name(), 	currEn,	l2w));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_leftWord3.name(), 	currEn,	l3w));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_rightWord1.name(), 	currEn,	rw));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_rightWord2.name(), 	currEn,	r2w));
//			featureList.add(this._param_g.toFeature(network,FEATYPE.entity_rightWord3.name(), 	currEn,	r3w));
//			
//			String[] pats = new String[]{"[A-Z][a-z]+", "[A-Z]", "[A-Z]+", "[A-Z]+[a-z]+[A-Z]+[a-z]+", ".*[0-9].*"};
//			Pattern r = null; 
//			Matcher m = null; 
//			for(int pt=0; pt<pats.length; pt++){
//				r = Pattern.compile(pats[pt]);
//				m = r.matcher(currWord);
//				switch(pt){
//				case 0:
//					if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.entity_type1.name(), currEn, "Type1"));
//					break;
//				case 1:
//					if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.entity_type2.name(), currEn, "Type1"));
//					break;
//				case 2:
//					if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.entity_type3.name(), currEn, "Type3"));
//					break;
//				case 3:
//					if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.entity_type4.name(), currEn, "Type4"));
//					break;
//				case 4:
//					if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.entity_type5.name(), currEn, "Type5"));
//					break;
//				}
//			}
			if(NetworkConfig.USE_NEURAL_FEATURES){
				featureList.add(this._param_g.toFeature(network, FEATYPE.neural_1.name(), Entity.get(eId).getForm(), lw.toLowerCase()+IN_SEP+
						currWord.toLowerCase()+IN_SEP+rw.toLowerCase()));
			}
		}
	}

	private void addPOSFeatures(ArrayList<Integer> featureList, Network network, Sentence sent, int pos, int tId){
		String currTag = tId==(Entity.ENTS.size()+Tag.TAGS.size())? "END":Tag.get(tId).getForm();
		String currWord = tId==(Entity.ENTS.size()+Tag.TAGS.size())? "END": sent.get(pos).getName();
		
		
		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_currWord.name(), 	currTag,	currWord));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_leftWord1.name(), 	currTag,	prevWord));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_leftWord2.name(), 	currTag,	p2w));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_leftWord3.name(), 	currTag,	p3w));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_rightWord1.name(), 	currTag,	nextWord));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_rightWord2.name(), 	currTag,	n2w));
//		featureList.add(this._param_g.toFeature(network,FEATYPE.tag_rightWord3.name(), 	currTag,	n3w));
		
		
//		String[] pats = new String[]{"[A-Z][a-z]+", "[A-Z]", "[A-Z]+", "[A-Z]+[a-z]+[A-Z]+[a-z]+", ".*[0-9].*"};
//		Pattern r = null; 
//		Matcher m = null; 
//		for(int pt=0; pt<pats.length; pt++){
//			r = Pattern.compile(pats[pt]);
//			m = r.matcher(currWord);
//			switch(pt){
//			case 0:
//				if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.tag_type1.name(), currTag, "Type1"));
//				break;
//			case 1:
//				if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.tag_type2.name(), currTag, "Type2"));
//				break;
//			case 2:
//				if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.tag_type3.name(), currTag, "Type3"));
//				break;
//			case 3:
//				if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.tag_type4.name(), currTag, "Type4"));
//				break;
//			case 4:
//				if(m.find()) featureList.add(this._param_g.toFeature(network,FEATYPE.tag_type5.name(), currTag, "Type5"));
//				break;
//			}
//		}
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
			featureList.add(this._param_g.toFeature(network, FEATYPE.neural_2.name(), currTag, currWord.toLowerCase() ));
		}
	}
	
	/**
	 * 
	 * @param featureList
	 * @param network
	 * @param sent
	 * @param pos
	 * @param paId
	 * @param parent_k
	 * @param children_k
	 * @param paTchildE: false means the current structure is NE structure.
	 */
	private void addJointFeatures(ArrayList<Integer> featureList, Network network, Sentence sent, int pos, int paId, int parent_k, int[] children_k, boolean paTchildE){
		TFNetwork tfnetwork = (TFNetwork)network;
		if(children_k.length!=1)
			throw new RuntimeException("The joint features should only have one children also");
		String currLabel = paTchildE? Tag.get(paId).getForm():Entity.get(paId).getForm();
		int jointFeatureIdx = -1; 
		//the joint feature here is always string (entityLabel, tagLabel);
		
		
		int[] arr = null;
		int nodeType = -1;
		if(!paTchildE){
			//current it's NE structure, need to refer to Tag node.
			nodeType = NODE_TYPES.TNODE.ordinal();
			for(int t=0;t<Tag.TAGS_INDEX.size();t++){
				String tag =  Tag.get(t).getForm();
				jointFeatureIdx = this._param_g.toFeature(network, FEATYPE.joint.name(), currLabel, tag);
				featureList.add(jointFeatureIdx);
				arr = new int[]{pos+1, nodeType, t, 0, 0};
				addDstNode(tfnetwork, jointFeatureIdx, parent_k, arr);
			}
			
		}else{
			//current it's POS structure, need to refer to Entity node
			nodeType = NODE_TYPES.ENODE.ordinal();
			for(int e=0;e<Entity.ENTS_INDEX.size(); e++){
				String entity = Entity.get(e).getForm();
				jointFeatureIdx = this._param_g.toFeature(network, FEATYPE.joint.name(), entity, currLabel);
				featureList.add(jointFeatureIdx);
				arr = new int[]{pos+1, nodeType, e, 0, 0};
//				System.out.println("unlabel:"+jointFeatureIdx);
				addDstNode(tfnetwork, jointFeatureIdx, parent_k, arr);
			}
		}
			
		
	}
	
	private void addDstNode(TFNetwork network, int jointFeatureIdx, int parent_k, int[] dstArr){
		long unlabeledDstNode = NetworkIDMapper.toHybridNodeID(dstArr);
		TFNetwork unlabeledNetwork = (TFNetwork)network.getUnlabeledNetwork();
		int unlabeledDstNodeIdx = Arrays.binarySearch(unlabeledNetwork.getAllNodes(), unlabeledDstNode);
//		System.out.println("The dst node idx in unlabeled network is:"+unlabeledDstNodeIdx);
		network.putJointFeature(parent_k, jointFeatureIdx, unlabeledDstNodeIdx);
	}
	
	
}
