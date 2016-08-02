/**
 * 
 */
package ext.svm_struct;

import static com.statnlp.commons.Utils.print;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import com.statnlp.commons.types.Instance;
import com.statnlp.example.linear_crf.Label;
import com.statnlp.example.linear_crf.LinearCRFFeatureManager;
import com.statnlp.example.linear_crf.LinearCRFInstance;
import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.FeatureArray;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.ModelType;
import com.statnlp.hybridnetworks.NetworkModel;

/**
 * The main class to run CRF++ with the same pipeline as other models<br>
 * This was used mainly to compare our implementation of LinearCRF with the one in CRF++
 */
public class SVMStruct {
	
	public static int INPUT_LEN = 0;
	
	public static void main(String[] args) throws Exception{
//		String timestamp = Calendar.getInstance().getTime().toString();
		String trainFilename = "data/train.data";
		String testFilename = "data/test.data";
		String modelFilename = "experiments/test/svmstruct.model";
		String resultFilename = "experiments/test/svmstruct.result";
		String logFilename = "experiments/test/svmstruct.log";
		String svmStructDir = "/Users/aldrian_muis/Documents/tools/SVM_hmm";
		double c = 4.0;
		
		int argIndex = 0;
		boolean shouldStop = false;
		while(argIndex < args.length && !shouldStop){
			switch(args[argIndex].substring(1)){
			case "trainPath":
				trainFilename = args[argIndex+1];
				argIndex += 2;
				break;
			case "testPath":
				testFilename = args[argIndex+1];
				argIndex += 2;
				break;
			case "modelPath":
				modelFilename = args[argIndex+1];
				argIndex += 2;
				break;
			case "resultPath":
				resultFilename = args[argIndex+1];
				argIndex += 2;
				break;
			case "logPath":
				logFilename = args[argIndex+1];
				argIndex += 2;
				break;
			case "C":
				c = Double.parseDouble(args[argIndex+1]);
				argIndex += 2;
				break;
			case "svmStructDir":
				svmStructDir = args[argIndex+1];
				argIndex += 2;
				break;
			case "h":
			case "help":
				printHelp();
				System.exit(0);
			case "-":
				shouldStop = true;
				argIndex += 1;
				break;
			default:
				throw new IllegalArgumentException("Unrecognized argument: "+args[argIndex]);
			}
		}
		
		ProcessBuilder processBuilder;
		PrintStream outstream = null;
		
		LinearCRFInstance[] trainInstances = null;
		LinearCRFInstance[] testInstances = null;
		if(trainFilename != null){
			trainInstances = readCoNLLData(new BufferedReader(new FileReader(trainFilename)), true, true, -1);
		}
		if(testFilename != null){
			testInstances = readCoNLLData(new BufferedReader(new FileReader(testFilename)), true, false, -1);
		}

		NetworkConfig.NUM_THREADS = 4;
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = false;
		NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.MODEL_TYPE = ModelType.CRF;
		
		String[] argsToFeatureManager = new String[args.length-argIndex];
		for(int i=argIndex; i<args.length; i++){
			argsToFeatureManager[i-argIndex] = args[i];
		}
		LinearCRFFeatureManager fm = new LinearCRFFeatureManager(new GlobalNetworkParam(), argsToFeatureManager);
		
		LinearCRFNetworkCompiler compiler = new LinearCRFNetworkCompiler();
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);
		
		PrintStream logStream = new PrintStream(logFilename);
		PrintStream[] outstreams = new PrintStream[]{logStream, System.out};
		
		if(trainFilename != null){
			try {
				model.train(trainInstances, 1);
			} catch (Exception e) {}
//			File tempTrain = File.createTempFile("svmstruct.", ".tmp", new File("experiments/test"));
//			tempTrain.deleteOnExit();
			File trainData = new File("experiments/test/svmstruct.dat");
			long start = System.currentTimeMillis();
			print("Converting training file into SVM Struct format", outstreams);
			outstream = new PrintStream(trainData);
			for(LinearCRFInstance instance: trainInstances){
				outstream.print(toSVMStructFormat(instance, fm, compiler));
			}
			outstream.close();
			long end = System.currentTimeMillis();
			print(String.format("Done in %.3fs\n", (end-start)/1000.0), outstreams);
			processBuilder = new ProcessBuilder(svmStructDir+"/svm_hmm_learn", "-c", c+"", "-v", "3", "-y", "3", trainData.getAbsolutePath(), modelFilename);
			processBuilder.redirectErrorStream(true);
			final Process learnProcess = processBuilder.start();
			start = System.currentTimeMillis();
			Thread outputThread = new Thread(new Runnable(){
				private Scanner outputReader = new Scanner(learnProcess.getInputStream());
				public void run(){
					while(outputReader.hasNextLine()){
						print(outputReader.nextLine(), outstreams);
					}
					outputReader.close();
				}
			});
			outputThread.start();
			Thread errorThread = new Thread(new Runnable(){
				private Scanner errorReader = new Scanner(learnProcess.getErrorStream());
				public void run(){
					while(errorReader.hasNextLine()){
						print(errorReader.nextLine(), outstreams);
					}
					errorReader.close();
				}
			});
			errorThread.start();
			learnProcess.waitFor();
			outputThread.join();
			errorThread.join();
			end = System.currentTimeMillis();
			print(String.format("Training done in %.3fs", (end-start)/1000.0), outstreams);
			try{
				writeFeaturesAndLabels(fm, modelFilename);
			} catch (FileNotFoundException e){
				throw e;
			}
		}
		
		if(testFilename != null){
//			File tempTest = File.createTempFile("svmstruct-", ".tmp", new File("experiments/test"));
//			tempTest.deleteOnExit();
			File testData = new File("experiments/test/svmstruct-test.dat");
			long start = System.currentTimeMillis();
			System.out.print("Converting test file into SVM Struct format");
			outstream = new PrintStream(testData);
			for(LinearCRFInstance instance: testInstances){
				outstream.print(toSVMStructFormat(instance, fm, compiler));
			}
			outstream.close();
			long end = System.currentTimeMillis();
			System.out.printf("Done in %.3fs\n", (end-start)/1000.0);
			processBuilder = new ProcessBuilder(svmStructDir+"/svm_hmm_classify", testData.getAbsolutePath(), modelFilename, resultFilename);
			processBuilder.redirectErrorStream(true);
			final Process testProcess = processBuilder.start();
			Thread outputThread = new Thread(new Runnable(){
				private Scanner outputReader = new Scanner(testProcess.getInputStream());
				public void run(){
					while(outputReader.hasNextLine()){
						print(outputReader.nextLine(), outstreams);
					}
				}
			});
			outputThread.start();
			Thread errorThread = new Thread(new Runnable(){
				private Scanner errorReader = new Scanner(testProcess.getErrorStream());
				public void run(){
					while(errorReader.hasNextLine()){
						print(errorReader.nextLine(), outstreams);
					}
				}
			});
			errorThread.start();
			testProcess.waitFor();
			outputThread.join();
			errorThread.join();
			
			LinearCRFInstance[] predictions = new LinearCRFInstance[testInstances.length];
			BufferedReader outputReader = new BufferedReader(new FileReader(resultFilename));
			for(int i=0; i<predictions.length; i++){
				LinearCRFInstance testInstance = testInstances[i];
				LinearCRFInstance predInstance = testInstance.duplicate();
				predInstance.input = testInstance.input;
				predInstance.output = testInstance.output;
				ArrayList<Label> prediction = new ArrayList<Label>();
				int len = testInstance.size();
				if(INPUT_LEN > 0){
					len = INPUT_LEN;
				}
				for(int wordIdx=0; wordIdx < len; wordIdx++){
					prediction.add(Label.get(Integer.parseInt(outputReader.readLine())-1));
				}
				predInstance.setPrediction(prediction);
				predictions[i] = predInstance;
			}
			outputReader.close();
			
			int corr = 0;
			int total = 0;
			int count = 0;
			for(Instance ins: predictions){
				LinearCRFInstance instance = (LinearCRFInstance)ins;
				ArrayList<Label> goldLabel = instance.getOutput();
				ArrayList<Label> actualLabel = instance.getPrediction();
				ArrayList<String[]> words = instance.getInput();
				int len = goldLabel.size();
				if(INPUT_LEN > 0){
					len = INPUT_LEN;
				}
				for(int i=0; i<len; i++){
					if(goldLabel.get(i).equals(actualLabel.get(i))){
						corr++;
					}
					total++;
					if(count < 3){
//						System.out.println(words.get(i)[0]+" "+words.get(i)[1]+" "+goldLabel.get(i).getId()+" "+actualLabel.get(i).getId());
						print(words.get(i)[0]+" "+goldLabel.get(i)+" "+actualLabel.get(i), outstreams);
					}
				}
				count++;
				if(count < 3){
					print("", outstreams);
				}
			}
			print(String.format("Correct/Total: %d/%d", corr, total), outstreams);
			print(String.format("Accuracy: %.2f%%", 100.0*corr/total), outstreams);
		}
	}
	
	private static LinearCRFInstance[] readCoNLLData(BufferedReader br, boolean withLabels, boolean isLabeled, int number) throws IOException{
		ArrayList<LinearCRFInstance> result = new ArrayList<LinearCRFInstance>();
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int instanceId = 1;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.length() == 0){
				LinearCRFInstance instance = new LinearCRFInstance(instanceId, 1, words, labels);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				if(result.size()==number) break;
				words = null;
				labels = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String[] features = line.substring(0, lastSpace).split(" ");
				words.add(features);
				if(withLabels){
					Label label = Label.get(line.substring(lastSpace+1));
					labels.add(label);
				}
			}
		}
		br.close();
		return result.toArray(new LinearCRFInstance[result.size()]);
	}
	
	private static String toSVMStructFormat(LinearCRFInstance instance, LinearCRFFeatureManager fm, LinearCRFNetworkCompiler compiler){
		StringBuilder result = new StringBuilder();
		int len = instance.size();
		if(INPUT_LEN > 0){
			len = INPUT_LEN;
		}
		for(int i=0; i<len; i++){
			instance.setUnlabeled();
			result.append((instance.getOutput().get(i).getId()+1)+" qid:"+instance.getInstanceId());
			Network network = compiler.compile(i, instance, new LocalNetworkParam(0, fm, 1));
			int parent_k = network.getNodeIndex(compiler.toNode(i, 0));
			int child_k;
			if(i > 0){
				child_k = network.getNodeIndex(compiler.toNode(i-1, 0));
			} else {
				child_k = network.getNodeIndex(compiler.toNode_leaf());
			}
			FeatureArray fa = fm.extract(network, parent_k, new int[]{child_k}, 0);
			ArrayList<Integer> featIds = new ArrayList<Integer>();
			for(int featId: fa.getCurrent()){
				if(featId >= 0){
					featIds.add(featId+1);
				}
			}
			Collections.sort(featIds);
			for(int featId: featIds){
				result.append(" "+featId+":1");
			}
			result.append(" #"+instance.getInput().get(i)[0]);
			result.append("\n");
		}
		return result.toString();
	}
	
	private static void writeFeaturesAndLabels(LinearCRFFeatureManager fm, String filename) throws IOException{
		Scanner modelReader = new Scanner(new File(filename));
		modelReader.nextLine(); // SVM-HMM Version V3.10
		modelReader.nextLine(); // kernel type
		modelReader.nextLine(); // kernel parameter -d
		modelReader.nextLine(); // kernel parameter -g
		modelReader.nextLine(); // kernel parameter -s
		modelReader.nextLine(); // kernel parameter -r
		modelReader.nextLine(); // kernel parameter -u
//		int maxFeatureIndex = modelReader.nextInt();
		modelReader.nextLine(); // highest feature index
		int numEmissions = modelReader.nextInt();
		modelReader.nextLine(); // number of emission features
		int numClasses = modelReader.nextInt();
		modelReader.nextLine(); // number of classes
		int orderTransition = modelReader.nextInt();
		modelReader.nextLine(); // HMM order of transitions
//		int orderEmission = modelReader.nextInt();
		modelReader.nextLine(); // HMM order of emissions
		modelReader.nextLine(); // loss function
		modelReader.nextLine(); // number of support vectors plus 1
		modelReader.nextLine(); // threshold b, each following line is a SV (starting with alpha*y)
		String supportVector = modelReader.nextLine();
		String[] features = supportVector.split(" ");
		HashMap<Integer, Double> weights = new HashMap<Integer, Double>();
		for(int i=1; i<features.length-1; i++){
			String[] tokens = features[i].split(":");
			int featId = Integer.parseInt(tokens[0]);
			double featWeight = Double.parseDouble(tokens[1]);
			weights.put(featId, featWeight);
		}
		modelReader.close();
		int emissionBaseFeatId = 1;
		for(int i=0; i<=orderTransition; i++){
			emissionBaseFeatId = emissionBaseFeatId*numClasses + 1;
		}
		PrintStream modelTextWriter = new PrintStream(filename+".features");
		GlobalNetworkParam paramG = fm.getParam_G();
		HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = paramG.getFeatureIntMap();
		for(String featureType: sorted(featureIntMap.keySet())){
			modelTextWriter.println(featureType);
			HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(featureType);
			for(int labelId: Label.LABELS_INDEX.keySet()){
				modelTextWriter.println("\t"+labelId);
				HashMap<String, Integer> inputMap = outputInputMap.get("-1");
				for(String input: sorted(inputMap.keySet())){
					int emissionFeatId = inputMap.get(input);
					int featureId = emissionBaseFeatId + (labelId+1)*numEmissions + (emissionFeatId+1);
					modelTextWriter.println("\t\t"+input+" "+String.format("%.16f", weights.getOrDefault(featureId, 0.0)));
				}
			}
		}
		modelTextWriter.println("TRANSITION");
		for(int labelId: Label.LABELS_INDEX.keySet()){
			for(int nextLabelId: Label.LABELS_INDEX.keySet()){
				int featureId = (labelId+1)*numClasses + (nextLabelId+1);
				modelTextWriter.println("\t"+labelId+" "+nextLabelId);
				modelTextWriter.println("\t\t"+String.format("%.16f", weights.getOrDefault(featureId, 0.0)));
			}
		}
		modelTextWriter.close();
	}
	
	private static List<String> sorted(Set<String> coll){
		List<String> result = new ArrayList<String>(coll);
		Collections.sort(result);
		return result;
	}
	
	private static void printHelp(){
		System.out.println("Options:\n"
				+ "-trainPath <trainPath>\n"
				+ "\tTake training file from <trainPath>. If not specified, no training will be performed\n"
				+ "-testPath <testPath>\n"
				+ "\tTake test file from <testPath>. If not specified, no testing will be performed\n"
				+ "-modelPath <modelPath>\n"
				+ "\tSerialize model to <modelPath>\n"
				+ "-resultPath <resultPath>\n"
				+ "\tPrint the result to <resultPath>\n"
				+ "-logPath <logPath>\n"
				+ "\tPrint log information to the specified file\n"
				+ "-C <value>\n"
				+ "\tSet the L2 regularization parameter weight to <value>. Default to 4.0\n"
				+ "-svmStructDir <dir>\n"
				+ "\tThe path to the directory containing svm_hmm_learn and svm_hmm_classify binaries\n"
				);
	}

}
