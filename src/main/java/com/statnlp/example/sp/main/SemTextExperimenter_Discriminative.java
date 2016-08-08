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
package com.statnlp.example.sp.main;

import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;

import com.statnlp.commons.ml.opt.OptimizerFactory;
import com.statnlp.commons.types.Instance;
import com.statnlp.example.sp.GeoqueryEvaluator;
import com.statnlp.example.sp.HybridGrammar;
import com.statnlp.example.sp.HybridGrammarReader;
import com.statnlp.example.sp.SemTextDataManager;
import com.statnlp.example.sp.SemTextFeatureManager_Discriminative;
import com.statnlp.example.sp.SemTextInstance;
import com.statnlp.example.sp.SemTextInstanceReader;
import com.statnlp.example.sp.SemTextNetworkCompiler;
import com.statnlp.example.sp.SemanticForest;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GenerativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.neural.NeuralConfigReader;

public class SemTextExperimenter_Discriminative {
	
	static boolean DEBUG = true;
	static boolean SKIP_TEST = false;
	static boolean PRINT_FEATS = false;
	
	public static void main(String args[]) throws Exception{
		
		System.err.println(SemTextExperimenter_Discriminative.class.getCanonicalName());
		
		String lang = args[1];
		String inst_filename = "data/geoquery/geoFunql-"+lang+".corpus";
		String init_filename = "data/geoquery/geoFunql-"+lang+".init.corpus";
		String g_filename = "data/hybridgrammar.txt";
		
		double adagrad_learningRate = 0.01;
		
		int modelIter = 0; // Integer.parseInt(args[3]);
		String modelPath = "";
		if (modelIter > 0) {
			modelPath = lang;
		}
		boolean isTrain = modelPath.equals("");
		
		String train_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/train-N600";//+args[1];
		
		// rhs: 20 train insts
		if (DEBUG) {
			train_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/train-N20";//+args[1];
		}
		String test_ids = "data/geoquery-2012-08-27/splits/split-880/run-0/fold-0/test";
		
		boolean isGeoquery = true;
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.NUM_THREADS = Integer.parseInt(args[0]);
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true; // true may change result
//		NetworkConfig._SEMANTIC_PARSING_NGRAM = Integer.parseInt(args[2]);
		
//		int numIterations = 100;//Integer.parseInt(args[3]);
		
		// rhs: iters
		int numIterations = 100;
		if (DEBUG) {
			numIterations = 10;
		}
		
//		NetworkConfig._SEMANTIC_FOREST_MAX_DEPTH = Integer.parseInt(args[4]);
		
		SemTextDataManager dm = new SemTextDataManager();
		
		ArrayList<SemTextInstance> inits = SemTextInstanceReader.readInit(init_filename, dm);
		ArrayList<SemTextInstance> insts_train = SemTextInstanceReader.read(inst_filename, dm, train_ids, true);
		ArrayList<SemTextInstance> insts_test = SemTextInstanceReader.read(inst_filename, dm, test_ids, false);

		// rhs: test same as train
//		ArrayList<SemTextInstance> insts_test = SemTextInstanceReader.read(inst_filename, dm, train_ids, false);
		
//		insts_test = insts_train;
		
		NetworkConfig.USE_NEURAL_FEATURES = true;
		NetworkConfig.REGULARIZE_NEURAL_FEATURES = false;
		
		int size = insts_train.size();
		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
			size += inits.size();
		}
		
		SemTextInstance train_instances[] = new SemTextInstance[size];
		for(int k = 0; k<insts_train.size(); k++){
			train_instances[k] = insts_train.get(k);
			train_instances[k].setInstanceId(k);
			train_instances[k].setLabeled();
		}
				
		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
			for(int k = 0; k<inits.size(); k++){
				train_instances[k+insts_train.size()] = inits.get(k);
				train_instances[k+insts_train.size()].setInstanceId(k+insts_train.size());
				train_instances[k+insts_train.size()].setLabeled();
			}
		}
		
		System.err.println("Read.."+train_instances.length+" instances.");
		
		HybridGrammar g = HybridGrammarReader.read(g_filename);
		
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			NeuralConfigReader.readConfig(args[2]);
		}
		
		SemanticForest forest_global = SemTextInstanceReader.toForest(dm);
		
		GlobalNetworkParam param_G;
		if (!isTrain) {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
            param_G = (GlobalNetworkParam) ois.readObject();
            ois.close();
		} else {
			param_G = new GlobalNetworkParam();
			if(NetworkConfig.USE_NEURAL_FEATURES){
				param_G =  new GlobalNetworkParam(OptimizerFactory.getGradientDescentFactoryUsingAdaGrad(adagrad_learningRate));
			}
		}
		
		SemTextFeatureManager_Discriminative fm = new SemTextFeatureManager_Discriminative(param_G, g, dm);
//		SemTextFeatureManager fm = new SemTextFeatureManager(new GlobalNetworkParam(), g, dm);
		
		SemTextNetworkCompiler compiler = new SemTextNetworkCompiler(g, forest_global, dm);
		
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler) : DiscriminativeNetworkModel.create(fm, compiler);
		
		if (isTrain) {
			model.train(train_instances, numIterations, lang);
		}
		
		if (PRINT_FEATS) {
			GlobalNetworkParam paramG = fm.getParam_G();
			System.out.println("Num features: "+paramG.countFeatures());
			System.out.println("Features:");
			HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = paramG.getFeatureIntMap();
			for(String featureType: (featureIntMap.keySet())){
				System.out.println(featureType);
			     HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(featureType);
			     for(String output: (outputInputMap.keySet())){
			    	 System.out.println("\t"+output);
			          HashMap<String, Integer> inputMap = outputInputMap.get(output);
			          for(String input: (inputMap.keySet())){
			               int featureId = inputMap.get(input);
			               System.out.println("\t\t"+input+" "+featureId+" "+fm.getParam_G().getWeight(featureId));
			          }
			     }
			}
		}
		
		// rhs: skip decoding
		if (SKIP_TEST) {
			System.exit(0);
		}
		
		SemTextInstance test_instances[];
		Instance[] output_instances_unlabeled;
		
		test_instances = new SemTextInstance[insts_test.size()];
		for(int k = 0; k<test_instances.length; k++){
			test_instances[k] = insts_test.get(k);
			test_instances[k].setUnlabeled();
		}
		output_instances_unlabeled = model.decode(test_instances);
		
		double total = output_instances_unlabeled.length;
		double corr = 0;
		
		GeoqueryEvaluator eval = new GeoqueryEvaluator();
		
		ArrayList<String> expts = new ArrayList<String>();
		ArrayList<String> preds = new ArrayList<String>();
		
		for(int k = 0; k<output_instances_unlabeled.length; k++){
			Instance output_inst_U = output_instances_unlabeled[k];
			boolean r = output_inst_U.getOutput().equals(output_inst_U.getPrediction());
			System.err.println(output_inst_U.getInstanceId()+":\t"+r);
			if(r){
				corr ++;
			}
			System.err.println("=INPUT=");
			System.err.println(output_inst_U.getInput());
			System.err.println("=OUTPUT=");
			System.err.println(output_inst_U.getOutput());
			System.err.println("=PREDICTION=");
			System.err.println(output_inst_U.getPrediction());
			
			String expt = eval.toGeoQuery((SemanticForest)output_inst_U.getOutput());
			String pred = eval.toGeoQuery((SemanticForest)output_inst_U.getPrediction());
			
			expts.add(expt);
			preds.add(pred);
			
			if(isGeoquery){
				System.err.println("output:\t"+expt);
				System.err.println("predic:\t"+pred);
			}
		}
		
		System.err.println("text accuracy="+corr/total+"="+corr+"/"+total);
		eval.eval(preds, expts);
		
	}
	
}