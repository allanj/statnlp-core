package com.statnlp.example;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.statnlp.commons.ml.opt.OptimizerFactory;
import com.statnlp.commons.types.Instance;
import com.statnlp.example.linear_ne.ECRFContinuousFeatureValueProvider;
import com.statnlp.example.linear_ne.ECRFEval;
import com.statnlp.example.linear_ne.ECRFFeatureManager;
import com.statnlp.example.linear_ne.ECRFInstance;
import com.statnlp.example.linear_ne.ECRFNetworkCompiler;
import com.statnlp.example.linear_ne.EReader;
import com.statnlp.example.linear_ne.Entity;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.FeatureValueProvider;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.ModelType;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.neural.BidirectionalLSTM;
import com.statnlp.neural.MultiLayerPerceptron;

public class LinearNEMain {
	
	public static boolean DEBUG = false;

	public static int trainNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = 100;
	public static int numThreads = 5;
	public static String MODEL = "ssvm";
	public static double adagrad_learningRate = 0.1;
	public static double l2 = 0.01;
	
	public static String trainPath = "nn-crf-interface/nlp-from-scratch/me/eng.train.conll";
	public static String testFile = "nn-crf-interface/nlp-from-scratch/me/eng.testb.conll";
//	public static String trainPath = "nn-crf-interface/nlp-from-scratch/debug/debug.train.txt";
//	public static String testFile = "nn-crf-interface/nlp-from-scratch/debug/debug.train.txt";
	public static String nerOut = "nn-crf-interface/nlp-from-scratch/me/output/ner_out.txt";
	public static String neural_config = "nn-crf-interface/neural_server/neural.debug.config";
	public static String neuralType = "mlp";
	public static boolean iobes = true;
	public static int gpuId = -1;
	public static String nnOptimizer = "lbfgs";
	
	public static void main(String[] args) throws IOException, InterruptedException{

		processArgs(args);
		System.err.println("[Info] trainingFile: "+trainPath);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		
		List<ECRFInstance> trainInstances = null;
		List<ECRFInstance> testInstances = null;
		
		
		trainInstances = EReader.readData(trainPath,true,trainNumber, "IOBES");
		testInstances = EReader.readData(testFile,false,testNumber,"IOB");
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		
		if (DEBUG) {
			NetworkConfig.RANDOM_INIT_WEIGHT = false;
			NetworkConfig.FEATURE_INIT_WEIGHT = 0.1;
		}
		
		List<FeatureValueProvider> fvps = new ArrayList<FeatureValueProvider>();
		if(NetworkConfig.USE_NEURAL_FEATURES){
//			gnp =  new GlobalNetworkParam(OptimizerFactory.getGradientDescentFactory());
			if (neuralType.equals("lstm")) {
				int hiddenSize = 100;
				String optimizer = nnOptimizer;
				boolean bidirection = true;
				fvps.add(new BidirectionalLSTM(hiddenSize, bidirection, optimizer, Entity.Entities.size(), gpuId));
			} else if (neuralType.equals("continuous")) {
				fvps.add(new ECRFContinuousFeatureValueProvider(2, Entity.Entities.size()));
			} else if (neuralType.equals("mlp")) {
				fvps.add(new MultiLayerPerceptron(neural_config, Entity.Entities.size()));
			} else {
				throw new RuntimeException("Unknown neural type: " + neuralType);
			}
		} 
		GlobalNetworkParam gnp = new GlobalNetworkParam(OptimizerFactory.getLBFGSFactory(), fvps);
		
		System.err.println("[Info] "+Entity.Entities.size()+" entities: "+Entity.Entities.toString());
		
		ECRFInstance all_instances[] = new ECRFInstance[trainInstances.size()+testInstances.size()];
        int i = 0;
        for(; i<trainInstances.size(); i++) {
            all_instances[i] = trainInstances.get(i);
        }
        int lastId = all_instances[i-1].getInstanceId();
        for(int j = 0; j<testInstances.size(); j++, i++) {
            all_instances[i] = testInstances.get(j);
            all_instances[i].setInstanceId(lastId+j+1);
            all_instances[i].setUnlabeled();
        }
		
		ECRFFeatureManager fa = new ECRFFeatureManager(gnp, neuralType, false);
		ECRFNetworkCompiler compiler = new ECRFNetworkCompiler(iobes);
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		ECRFInstance[] ecrfs = trainInstances.toArray(new ECRFInstance[trainInstances.size()]);
		model.train(ecrfs, numIteration);
		Instance[] predictions = model.decode(testInstances.toArray(new ECRFInstance[testInstances.size()]));
		ECRFEval.evalNER(predictions, nerOut);
	}

	
	
	public static void processArgs(String[] args){
		if(args[0].equals("-h") || args[0].equals("help") || args[0].equals("-help") ){
			System.err.println("Linear-Chain CRF Version: Joint DEPENDENCY PARSING and Entity Recognition TASK: ");
			System.err.println("\t usage: java -jar dpe.jar -trainNum -1 -testNum -1 -thread 5 -iter 100 -pipe true");
			System.err.println("\t put numTrainInsts/numTestInsts = -1 if you want to use all the training/testing instances");
			System.exit(0);
		}else{
			for(int i=0;i<args.length;i=i+2){
				switch(args[i]){
					case "-trainNum": trainNumber = Integer.valueOf(args[i+1]); break;   //default: all 
					case "-testNum": testNumber = Integer.valueOf(args[i+1]); break;    //default:all
					case "-iter": numIteration = Integer.valueOf(args[i+1]); break;   //default:100;
					case "-thread": numThreads = Integer.valueOf(args[i+1]); break;   //default:5
					case "-testFile": testFile = args[i+1]; break;        
					case "-windows":ECRFEval.windows = true; break;            //default: false (is using windows system to run the evaluation script)
					case "-batch": NetworkConfig.USE_BATCH_TRAINING = true;
									NetworkConfig.BATCH_SIZE = Integer.valueOf(args[i+1]); break;
					case "-model": NetworkConfig.MODEL_TYPE = args[i+1].equals("crf")? ModelType.CRF:ModelType.SSVM;   break;
					case "-neural": if(args[i+1].equals("mlp") || args[i+1].equals("lstm")|| args[i+1].equals("continuous")){ 
											NetworkConfig.USE_NEURAL_FEATURES = true;
											neuralType = args[i+1]; //by default optim_neural is false.
											NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
											NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
									}
									break;
					case "-optimNeural": 
						NetworkConfig.OPTIMIZE_NEURAL = args[i+1].equals("true") ? true : false; //optimize the neural features or not
						if (!NetworkConfig.OPTIMIZE_NEURAL) {
							nnOptimizer = args[i+2];
							i++;
						}break;
					case "-gpuid": gpuId = Integer.valueOf(args[i+1]); break;
					case "-reg": l2 = Double.valueOf(args[i+1]);  break;
					case "-lr": adagrad_learningRate = Double.valueOf(args[i+1]); break;
					case "-backend": NetworkConfig.NEURAL_BACKEND = args[i+1]; break;
					case "-os": NetworkConfig.OS = args[i+1]; break; // for Lua native lib, "osx" or "linux" 
					default: System.err.println("Invalid arguments "+args[i]+", please check usage."); System.exit(0);
				}
			}
			System.err.println("[Info] trainNum: "+trainNumber);
			System.err.println("[Info] testNum: "+testNumber);
			System.err.println("[Info] numIter: "+numIteration);
			System.err.println("[Info] numThreads: "+numThreads);
			System.err.println("[Info] Regularization Parameter: "+l2);
		}
	}
}
