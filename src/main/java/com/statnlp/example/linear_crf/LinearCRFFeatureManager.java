/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * 
 */
package com.statnlp.example.linear_crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import com.statnlp.commons.types.Label;
import com.statnlp.commons.types.LinearInstance;
import com.statnlp.example.base.BaseNetwork;
import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkIDMapper;
import com.statnlp.util.Pipeline;
import com.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import com.statnlp.util.instance_parser.InstanceParser;

/**
 * @author Aldrian Obaja (aldrianobaja.m@gmail.com)
 */
public class LinearCRFFeatureManager extends FeatureManager{

	private static final long serialVersionUID = -4880581521293400351L;
	
	private static final boolean CHEAT = false;
	
	public int wordHalfWindowSize = 1;
	public int posHalfWindowSize = -1;
	public boolean productWithOutput = true;
	public Map<Integer, Label> labels;
	public Map<FeatureType, Boolean> featureTypes;
	
	public enum FeatureType {
		WORD(true),
		WORD_BIGRAM(false),
		TAG(false),
		TAG_BIGRAM(false),
		TRANSITION(true),
		LABEL(false),
		neural(false),
		;
		
		private boolean enabledByDefault;
		
		private FeatureType(){
			this(true);
		}
		
		private FeatureType(boolean enabledByDefault){
			this.enabledByDefault = enabledByDefault;
		}
		
		public boolean enabledByDefault(){
			return enabledByDefault;
		}
		
		public boolean disabledByDefault(){
			return !enabledByDefault;
		}
		
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g) {
		this(param_g, (InstanceParser)null, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, String[] args) {
		this(param_g, (InstanceParser)null, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, LinearCRFConfig config) {
		this(param_g, (InstanceParser)null, config);
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels) {
		this(param_g, labels, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels, String[] args){
		this(param_g, labels, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, Map<Integer, Label> labels, LinearCRFConfig config) {
		this(param_g, (InstanceParser)null, config);
		this.labels = labels;
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser) {
		this(param_g, instanceParser, new LinearCRFConfig());
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser, String[] args){
		this(param_g, instanceParser, new LinearCRFConfig(args));
	}
	
	public LinearCRFFeatureManager(GlobalNetworkParam param_g, InstanceParser instanceParser, LinearCRFConfig config) {
		super(param_g, instanceParser);
		if(instanceParser != null){
			this.labels = ((DelimiterBasedInstanceParser)instanceParser).LABELS_INDEX;
		}
		wordHalfWindowSize = config.wordHalfWindowSize;
		posHalfWindowSize = config.posHalfWindowSize;
		productWithOutput = config.productWithOutput;
		featureTypes = new HashMap<FeatureType, Boolean>();
		if(config.features != null){
			for(FeatureType featureType: FeatureType.values()){
				disable(featureType);
			}
			for(String feat: config.features){
				enable(FeatureType.valueOf(feat.toUpperCase()));
			}
		}
	}

	/**
	 * Enables the specified feature type.
	 * @param featureType
	 */
	public void enable(FeatureType featureType){
		featureTypes.put(featureType, true);
	}
	
	/**
	 * Disables the specified feature type.
	 * @param featureType
	 */
	public void disable(FeatureType featureType){
		featureTypes.put(featureType, false);
	}
	
	/**
	 * Returns whether the specified feature type is enabled.
	 * @param featureType
	 * @return
	 */
	public boolean isEnabled(FeatureType featureType){
		return featureTypes.get(featureType);
	}
	
	public LinearCRFFeatureManager(Pipeline<?> pipeline){
		this(pipeline.param, pipeline.instanceParser);
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		GlobalNetworkParam param_g = this._param_g;
		
		BaseNetwork net = (BaseNetwork)network;
		
		@SuppressWarnings("unchecked")
		LinearInstance<String> instance = (LinearInstance<String>)net.getInstance();
		
		ArrayList<String[]> input = (ArrayList<String[]>)instance.getInput();
		
		long curNode = net.getNode(parent_k);
		int[] arr = NetworkIDMapper.toHybridNodeArray(curNode);
		
		int pos = arr[0]-1;
		int tag_id = arr[1];
		int nodeType = arr[4];
		if(!productWithOutput){
			tag_id = -1;
		}
		
		if(nodeType == NODE_TYPES.ROOT.ordinal()){
			return FeatureArray.EMPTY;
		}
		
		if(nodeType == NODE_TYPES.LEAF.ordinal()){
			return FeatureArray.EMPTY;
		}
		
		//long childNode = network.getNode(children_k[0]);
		int child_tag_id = network.getNodeArray(children_k[0])[1];
		int childNodeType = network.getNodeArray(children_k[0])[4];
		
		int labelSize = labels.size();

		if(childNodeType == NODE_TYPES.LEAF.ordinal()){
			child_tag_id = labelSize;
		}
		
		if(CHEAT){
			return new FeatureArray(new int[]{param_g.toFeature(net, "CHEAT", tag_id+"", Math.abs(instance.getInstanceId())+" "+pos+" "+child_tag_id)});
		}
		String word = input.get(pos)[0];
		boolean firstIsUpper = word.substring(0,1).toUpperCase().equals(word.substring(0,1));
		boolean allIsUpper = word.toUpperCase().equals(word);
		int length = Math.min(word.length(), 6);
		boolean containsPeriod = word.contains(".");
		
		String label = tag_id == labelSize ? "O" : this.labels.get(tag_id).getForm();
		String childLabel = child_tag_id == labelSize ? "O" : this.labels.get(child_tag_id).getForm();
		
		if(!label.startsWith("O")){
			label = label.substring(2);
		}
		if(!childLabel.startsWith("O")){
			childLabel = childLabel.substring(2);
		}

		ArrayList<Integer> features = new ArrayList<Integer>();
		
		features.add(param_g.toFeature(network, "BIAS", "", ""));
		features.add(param_g.toFeature(network, "BIAS_TAG", label, ""));

		features.add(param_g.toFeature(network, "FIRSTISUPPER", label+"", ""+firstIsUpper));
		features.add(param_g.toFeature(network, "ALLISUPPER", label+"", ""+allIsUpper));
		features.add(param_g.toFeature(network, "PERIOD", label+"", ""+containsPeriod));
		features.add(param_g.toFeature(network, "LENGTH", label+"", ""+length));
		features.add(param_g.toFeature(network, "LENGTH_ALLISUPPER", label+"", ""+length+"_"+allIsUpper));
		features.add(param_g.toFeature(network, "LENGTH_FIRSTISUPPER", label+"", ""+length+"_"+firstIsUpper));
		features.add(param_g.toFeature(network, "LENGTH_PERIOD", label+"", ""+length+"_"+containsPeriod));
		FeatureArray featureArr = createFeatureArray(network, features);
		featureArr = createFeatureArray(network, new int[]{param_g.toFeature(network, "TRANS", childLabel+"_"+label, "")}, featureArr);
		return featureArr;
	}

}
