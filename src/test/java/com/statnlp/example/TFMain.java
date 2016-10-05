package com.statnlp.example;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import com.statnlp.commons.types.Instance;
import com.statnlp.example.fcrf.Entity;
import com.statnlp.example.fcrf.GRMMFeatureManager;
import com.statnlp.example.fcrf.TFConfig;
import com.statnlp.example.fcrf.TFEval;
import com.statnlp.example.fcrf.TFInstance;
import com.statnlp.example.fcrf.TFNetworkCompiler;
import com.statnlp.example.fcrf.TFReader;
import com.statnlp.example.fcrf.Tag;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.InferenceType;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.neural.NeuralConfigReader;

public class TFMain {

	public static String[] entities; 
	public static int trainNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = -100;
	public static int numThreads = -100;
	public static String trainFile = "";
	public static String testFile = "";
	public static String nerOut;
	public static String posOut;
	public static String[] selectedEntities = {"person","organization","gpe","MISC"};
	public static String neural_config = "nn-crf-interface/neural_server/neural.config";
	
	public static void main(String[] args) throws IOException, InterruptedException{
		// TODO Auto-generated method stub
		
		trainNumber = 80;
		testNumber = 2;
		numThreads = 5;
		numIteration = 200;
		processArgs(args);
		
		trainFile = TFConfig.CONLL_train;
		testFile = TFConfig.CONLL_test;
		nerOut = TFConfig.nerOut;
		posOut = TFConfig.posOut;
		
	
		
		List<TFInstance> trainInstances = null;
		List<TFInstance> testInstances = null;
		/***********DEBUG*****************/
		trainFile = "data/dat/conll2000.train1k.txt";
		String trainSrcFile = "data/dat/conll1000train.txt";
		trainNumber = -1;
		testFile = "data/dat/conll2000.test1k.txt";;
		String testSrcFile = "data/dat/conll1000test.txt";
		testNumber = -1;
		/***************************/
		
		System.err.println("[Info] trainingFile: "+TFConfig.CONLL_train);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		System.err.println("[Info] posOut: "+posOut);
		
//		trainInstances = TFReader.readCONLLData(trainFile, true, trainNumber);
//		testInstances = TFReader.readCONLLData(testFile, false, testNumber);
		
		trainInstances = TFReader.readGRMMDataAndWord(trainFile, true, trainNumber, trainSrcFile);
		testInstances = TFReader.readGRMMDataAndWord(testFile, false, testNumber, testSrcFile);
		
		System.err.println("entity size:"+Entity.ENTS_INDEX.toString());
		System.err.println("tag size:"+Tag.TAGS.size());
		System.err.println("tag size:"+Tag.TAGS_INDEX.toString());
//		Formatter.ner2Text(trainInstances, "data/testRandom2.txt");
//		System.exit(0);
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = TFConfig.l2val;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = false;
		NetworkConfig.INFERENCE = InferenceType.MEAN_FIELD;
		
		
		/***Neural network Configuration**/
		NetworkConfig.USE_NEURAL_FEATURES = true; 
		if(NetworkConfig.USE_NEURAL_FEATURES)
			NeuralConfigReader.readConfig(neural_config);
		NetworkConfig.OPTIMIZE_NEURAL = false;  //false: optimize in neural network
		NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
		/****/
		
//		TFFeatureManager fa = new TFFeatureManager(new GlobalNetworkParam());
		GRMMFeatureManager fa = new GRMMFeatureManager(new GlobalNetworkParam());
		TFNetworkCompiler compiler = new TFNetworkCompiler();
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		TFInstance[] ecrfs = trainInstances.toArray(new TFInstance[trainInstances.size()]);
		model.train(ecrfs, numIteration);
		Instance[] predictions = model.decode(testInstances.toArray(new TFInstance[testInstances.size()]));
//		TFEval.evalNER(predictions, nerOut);
		TFEval.evalSingleE(predictions);
		TFEval.evalPOS(predictions, posOut);
		TFEval.evalSingleJoint(predictions);
	}

	
	
	public static void processArgs(String[] args){
		if(args.length>1 && (args[0].equals("-h") || args[0].equals("help") || args[0].equals("-help") )){
			System.err.println("Linear-Chain CRF Version: Joint DEPENDENCY PARSING and Entity Recognition TASK: ");
			System.err.println("\t usage: java -jar dpe.jar -trainNum -1 -testNum -1 -thread 5 -iter 100 -pipe true");
			System.err.println("\t put numTrainInsts/numTestInsts = -1 if you want to use all the training/testing instances");
			System.exit(0);
		}else{
			for(int i=0;i<args.length;i=i+2){
				switch(args[i]){
					case "-trainNum": trainNumber = Integer.valueOf(args[i+1]); break;
					case "-testNum": testNumber = Integer.valueOf(args[i+1]); break;
					case "-iter": numIteration = Integer.valueOf(args[i+1]); break;
					case "-thread": numThreads = Integer.valueOf(args[i+1]); break;
					case "-ent": selectedEntities = args[i+1].split(","); break;
					case "-testFile": testFile = args[i+1]; break;
					case "-reg": TFConfig.l2val = Double.valueOf(args[i+1]); break;
					case "-windows":TFConfig.windows = true; break;
					case "-mfround":NetworkConfig.MF_ROUND = Integer.valueOf(args[i+1]); break;
					default: System.err.println("Invalid arguments, please check usage."); System.exit(0);
				}
			}
			System.err.println("[Info] trainNum: "+trainNumber);
			System.err.println("[Info] testNum: "+testNumber);
			System.err.println("[Info] numIter: "+numIteration);
			System.err.println("[Info] numThreads: "+numThreads);
			System.err.println("[Info] Selected Entities: "+Arrays.toString(selectedEntities));
			System.err.println("[Info] Regularization Parameter: "+TFConfig.l2val);	
		}
	}
}
