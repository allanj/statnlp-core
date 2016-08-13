package com.statnlp.example;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;

import com.statnlp.commons.ml.opt.OptimizerFactory;
import com.statnlp.commons.types.Instance;
import com.statnlp.example.linear_ne.ECRFEval;
import com.statnlp.example.linear_ne.ECRFFeatureManager;
import com.statnlp.example.linear_ne.ECRFInstance;
import com.statnlp.example.linear_ne.ECRFNetworkCompiler;
import com.statnlp.example.linear_ne.EConfig;
import com.statnlp.example.linear_ne.EReader;
import com.statnlp.example.linear_ne.Entity;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.ModelType;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.neural.NeuralConfig;
import com.statnlp.neural.NeuralConfigReader;

public class LinearNEMain {

	public static int trainNumber = -100;
	public static int testNumber = -100;
	public static int numIteration = 5000;
	public static int numThreads = 5;
	public static String MODEL = "ssvm";
	public static double adagrad_learningRate = 0.01;
	public static double l2 = 0.0;
	
	public static String trainPath = "nn-crf-interface/nlp-from-scratch/me/eng.train.conll";
	public static String testFile = "nn-crf-interface/nlp-from-scratch/me/eng.testb.conll";
//	public static String trainPath = "nn-crf-interface/nlp-from-scratch/debug/debug.train.txt";
//	public static String testFile = "nn-crf-interface/nlp-from-scratch/debug/debug.train.txt";
	public static String nerOut = "nn-crf-interface/nlp-from-scratch/me/output/ner_out.txt";
	public static String neural_config = "nn-crf-interface/neural_server/neural.config";
	
	public static String loadModel = "";
	public static boolean testOnTrain = false;
	public static boolean printFeats = false;
	public static boolean printNeuralFeats = false;
	public static boolean loadNeuralWeights = false;
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException{
		
		processArgs(args);
		System.err.println("[Info] trainingFile: "+trainPath);
		System.err.println("[Info] testFile: "+testFile);
		System.err.println("[Info] nerOut: "+nerOut);
		
		List<ECRFInstance> trainInstances = null;
		List<ECRFInstance> testInstances = null;
		List<ECRFInstance> testInstancesClone = null;
		
		trainInstances = EReader.readData(trainPath,true,trainNumber, "IOB");
		if (testOnTrain) {
			testInstances = EReader.readData(trainPath,false,trainNumber,"IOB");
			testInstancesClone = EReader.readData(trainPath,false,trainNumber,"IOB");
		} else {
			testInstances = EReader.readData(testFile,false,testNumber,"IOB");
			testInstancesClone = EReader.readData(testFile,false,testNumber,"IOB");
		}
		
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		
		boolean isTrain = loadModel.equals("");
		
		GlobalNetworkParam gnp;
		if (isTrain) {
			gnp = new GlobalNetworkParam(OptimizerFactory.getLBFGSFactory());
		} else {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(loadModel));
            gnp = (GlobalNetworkParam) ois.readObject();
            ois.close();
		}
		
		if(NetworkConfig.USE_NEURAL_FEATURES){
			NeuralConfigReader.readConfig(neural_config);
			//gnp =  new GlobalNetworkParam(OptimizerFactory.getGradientDescentFactoryUsingAdaGrad(adagrad_learningRate));
		}
		
		
		System.err.println("[Info] "+Entity.Entities.size()+" entities: "+Entity.Entities.toString());
		
		ECRFFeatureManager fa = new ECRFFeatureManager(gnp);
		ECRFNetworkCompiler compiler = new ECRFNetworkCompiler();
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		ECRFInstance[] ecrfs = trainInstances.toArray(new ECRFInstance[trainInstances.size()]);
		ECRFInstance[] test_ecrfs = testInstancesClone.toArray(new ECRFInstance[testInstances.size()]);
		
//		NetworkConfig.SAVE_MODEL_AFTER_ITER = -1;
		
		if (isTrain) {
			int trainSize = trainInstances.size();
			int testSize = testInstances.size();
			ECRFInstance[] allInsts = new ECRFInstance[trainSize+testSize];
			int i = 0;
			for(; i < trainSize; i++) {
				allInsts[i] = ecrfs[i];
			}
			int lastId = allInsts[i-1].getInstanceId();
			for(int j = 0; j < testSize; i++, j++) { // this part is "unlabeled"
				allInsts[i] = test_ecrfs[j];
				allInsts[i].setInstanceId(lastId+j+1);
			}
			model.train(allInsts, trainSize, numIteration, "ner");
			//model.train(ecrfs, trainSize, numIteration, "ner");
		}
		
		if (printNeuralFeats) {
			PrintWriter pw = new PrintWriter(new File("model/neural.txt"));
			HashSet<String> inputSet = new HashSet<String>();
			GlobalNetworkParam paramG = fa.getParam_G();
			HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = paramG.getFeatureIntMap();
			HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get("neural");
			for(String output: (outputInputMap.keySet())){
				HashMap<String, Integer> inputMap = outputInputMap.get(output);
				for(String input: (inputMap.keySet())){
					if (!inputSet.contains(input)) {
	            	   String[] fields = input.split(NeuralConfig.OUT_SEP);
	            	   String out = "";
	            	   for (String f : fields) {
	            		   String[] ele = f.split(NeuralConfig.IN_SEP);
	            		   for (String e : ele) {
	            			   out += (Integer.parseInt(e)+1) + " ";
	            		   }
	            	   }
	            	   pw.write(out+"\n");
		            }
		            inputSet.add(input);
				}
			}
			pw.close();
		}
		
		if (loadNeuralWeights) {
			int numLabel = 9;
			HashMap<String,Integer> mapper = new HashMap<String,Integer>();
			Scanner sc = new Scanner(new File("nn-crf-interface/nlp-from-scratch/senna-torch/senna/hash/ner9.lst"));
			for(int i = 0; i < numLabel; i++) {
				mapper.put(sc.nextLine().trim(), i);
			}
			sc.close();
			
			GlobalNetworkParam paramG = fa.getParam_G();
			HashMap<String, HashMap<String, Integer>> outputInputMap = paramG.getFeatureIntMap().get("neural");
			
			sc = new Scanner(new File("nn-crf-interface/nlp-from-scratch/nlpfromscratch/scores.txt"));
			while(sc.hasNextLine()) {
				String input = "";
				boolean first = true;
				for(int i = 0; i < 5; i++) {
					if(!first) input += "#IN#";
					input += ""+(sc.nextInt()-1);
					first = false;
				}
				input += "#OUT#";
				first = true;
				for(int i = 0; i < 5; i++) {
					if(!first) input += "#IN#";
					input += ""+(sc.nextInt()-1);
					first = false;
				}
				double[] ws = new double[numLabel];
				for(int i = 0; i < numLabel; i++) {
					ws[i] = sc.nextDouble();
				}
				for(String output: outputInputMap.keySet()) {
					HashMap<String, Integer> inputMap = outputInputMap.get(output);
					if(inputMap == null) {
						System.out.println("not found mapping for OUTPUT: "+output);
						continue;
					}
					Integer featIdx = inputMap.get(input);
					if(featIdx == null) {
						System.out.println("not found mapping for INPUT-OUTPUT: "+input+", "+output);
						continue;
					}
					paramG.overRideWeight(featIdx, ws[mapper.get(output)]);
				}
				sc.nextLine();
			}
			sc.close();
		}
		
		if (printFeats) {
			GlobalNetworkParam paramG = fa.getParam_G();
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
			               System.out.println("\t\t"+input.replaceAll("#IN#", " ").replaceAll("#OUT#", " ")+" "+featureId+" "+fa.getParam_G().getWeight(featureId));
			          }
			     }
			}
		}
		
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
					case "-windows":EConfig.windows = true; break;            //default: false (is using windows system to run the evaluation script)
					case "-batch": NetworkConfig.USE_BATCH_TRAINING = true;
									NetworkConfig.BATCH_SIZE = Integer.valueOf(args[i+1]);
									NetworkConfig.RANDOM_BATCH = false; break;
					case "-model": NetworkConfig.MODEL_TYPE = args[i+1].equals("crf")? ModelType.CRF:ModelType.SSVM;   break;
					case "-neural": if(args[i+1].equals("true")){
										NetworkConfig.USE_NEURAL_FEATURES = true;
										NetworkConfig.OPTIMIZE_NEURAL = true;
										NetworkConfig.IS_INDEXED_NEURAL_FEATURES = true;
										NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
									}
									break;
					case "-reg": l2 = Double.valueOf(args[i+1]);  break;
					case "-lr": adagrad_learningRate = Double.valueOf(args[i+1]); break;
					case "-saveIter": NetworkConfig.SAVE_MODEL_AFTER_ITER = Integer.valueOf(args[i+1]); break;
					case "-loadModel": loadModel = args[i+1]; break;
					case "-testOnTrain": testOnTrain = true; break;
					case "-printFeats": printFeats = true; break;
					case "-printNeuralFeats": printNeuralFeats = true; break;
					case "-loadNeuralWeights": loadNeuralWeights = true; break;
					default: System.err.println("Invalid arguments, please check usage."); System.exit(0);
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
