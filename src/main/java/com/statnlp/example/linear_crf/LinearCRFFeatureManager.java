/** Statistical Natural Language Processing System
    Copyright (C) 2014-2015  Lu, Wei

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

import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler.NODE_TYPES;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkIDMapper;

/**
 * @author wei_lu
 *
 */
public class LinearCRFFeatureManager extends FeatureManager{

	private static final long serialVersionUID = -4880581521293400351L;
	
	public int wordHalfWindowSize = 1;
	public int posHalfWindowSize = -1;
	public boolean useWordBigram = true;
	public boolean usePOSBigram = false;
	
	public enum FeatureType {
		WORD,
		WORD_BIGRAM(false),
		TAG(false),
		TAG_BIGRAM(false),
		TRANSITION,
		;
		
		private boolean isEnabled;
		
		private FeatureType(){
			this(true);
		}
		
		private FeatureType(boolean enabled){
			this.isEnabled = enabled;
		}
		
		public void enable(){
			this.isEnabled = true;
		}
		
		public void disable(){
			this.isEnabled = false;
		}
		
		public boolean enabled(){
			return isEnabled;
		}
		
		public boolean disabled(){
			return !isEnabled;
		}
	}
	
	/**
	 * @param param_g
	 */
	public LinearCRFFeatureManager(GlobalNetworkParam param_g) {
		super(param_g);
	}

	@Override
	protected FeatureArray extract_helper(Network network, int parent_k, int[] children_k) {
		
		LinearCRFNetwork net = (LinearCRFNetwork)network;
		
		LinearCRFInstance instance = (LinearCRFInstance)net.getInstance();
		int size = instance.size();
		
		ArrayList<String[]> input = instance.getInput();
		
		long curNode = net.getNode(parent_k);
		int[] arr = NetworkIDMapper.toHybridNodeArray(curNode);
		
		int pos = arr[0]-1;
		int tag_id = arr[1];
		int nodeType = arr[4];
		
		if(nodeType == NODE_TYPES.LEAF.ordinal()){
			return FeatureArray.EMPTY;
		}
		
		int child_tag_id = network.getNodeArray(children_k[0])[1];
		
		GlobalNetworkParam param_g = this._param_g;

		FeatureArray features = new FeatureArray(new int[0]);
		// Word window features
		if(FeatureType.WORD.enabled()){
			int wordWindowSize = wordHalfWindowSize*2+1;
			if(wordWindowSize < 0){
				wordWindowSize = 0;
			}
			int[] wordWindowFeatures = new int[wordWindowSize];
			for(int i=0; i<wordWindowFeatures.length; i++){
				String word = "***";
				int relIdx = i-wordHalfWindowSize;
				int idx = pos + relIdx;
				if(idx >= 0 && idx < size){
					word = input.get(idx)[0];
				}
				if(idx > pos) continue; // Only consider the left window
				wordWindowFeatures[i] = param_g.toFeature(network, FeatureType.WORD+":"+relIdx, tag_id+"", word);
			}
			FeatureArray wordFeatures = new FeatureArray(wordWindowFeatures, features);
			features = wordFeatures;
		}
		
		// POS tag window features
		if(FeatureType.TAG.enabled()){
			int posWindowSize = posHalfWindowSize*2+1;
			if(posWindowSize < 0){
				posWindowSize = 0;
			}
			int[] posWindowFeatures = new int[posWindowSize];
			for(int i=0; i<posWindowFeatures.length; i++){
				String postag = "***";
				int relIdx = i-posHalfWindowSize;
				int idx = pos + relIdx;
				if(idx >= 0 && idx < size){
					postag = input.get(idx)[1];
				}
				posWindowFeatures[i] = param_g.toFeature(network, FeatureType.TAG+":"+relIdx, tag_id+"", postag);
			}
			FeatureArray posFeatures = new FeatureArray(posWindowFeatures, features);
			features = posFeatures;
		}
		
		// Word bigram features
		if(FeatureType.WORD_BIGRAM.enabled()){
			int[] bigramFeatures = new int[2];
			for(int i=0; i<2; i++){
				String bigram = "";
				for(int j=0; j<2; j++){
					int idx = pos+i+j-1;
					if(idx >=0 && idx < size){
						bigram += input.get(idx)[0];
					} else {
						bigram += "***";
					}
					if(j==0){
						bigram += " ";
					}
				}
				bigramFeatures[i] = param_g.toFeature(network, FeatureType.WORD_BIGRAM+":"+i, tag_id+"", bigram);
			}
			features = new FeatureArray(bigramFeatures, features);
		}
		
		// POS tag bigram features
		if(FeatureType.TAG_BIGRAM.enabled()){
			int[] bigramFeatures = new int[2];
			for(int i=0; i<2; i++){
				String bigram = "";
				for(int j=0; j<2; j++){
					int idx = pos+i+j-1;
					if(idx >=0 && idx < size){
						bigram += input.get(idx)[1];
					} else {
						bigram += "***";
					}
					if(j==0){
						bigram += " ";
					}
				}
				bigramFeatures[i] = param_g.toFeature(network, FeatureType.TAG_BIGRAM+":"+i, tag_id+"", bigram);
			}
			features = new FeatureArray(bigramFeatures, features);
		}
		
		// Label transition feature
		if(FeatureType.TRANSITION.enabled()){
			int transitionFeature = param_g.toFeature(network, FeatureType.TRANSITION.name(), tag_id+"", child_tag_id+" "+tag_id);
			features = new FeatureArray(new int[]{transitionFeature}, features);
		}
		
		return features;
	}

}
