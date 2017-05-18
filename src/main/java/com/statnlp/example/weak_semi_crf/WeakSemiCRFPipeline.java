/**
 * 
 */
package com.statnlp.example.weak_semi_crf;

import com.statnlp.util.GenericPipeline;
import com.statnlp.util.instance_parser.InstanceParser;

public class WeakSemiCRFPipeline extends GenericPipeline {
	
	public InstanceParser initInstanceParser(){
		if(instanceParser != null){
			return instanceParser;
		}
		return new WeakSemiCRFInstanceParser(this);
	}
	
	public static void main(String[] args){
		new WeakSemiCRFPipeline().parseArgs(args).execute();
	}
}
