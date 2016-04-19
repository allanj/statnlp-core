package com.statnlp.example;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import com.statnlp.commons.types.Instance;
import com.statnlp.example.linear_crf.Label;
import com.statnlp.example.linear_crf.LinearCRFFeatureManager;
import com.statnlp.example.linear_crf.LinearCRFInstance;
import com.statnlp.example.linear_crf.LinearCRFNetworkCompiler;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;

public class LinearCRFMain {
	
	public static ArrayList<Label> allLabels;
	
	public static void main(String args[]) throws IOException, InterruptedException{
		
//		String lang = args[1];
		
		String inst_filename = "data/train.txt";
		String test_filename = "data/test.txt";
		
		int numTrain = 200;
		LinearCRFInstance[] trainInstances = readCoNLLData(inst_filename, true, true, numTrain);
		LinearCRFInstance[] testInstances = readCoNLLData(test_filename, true, false);
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig._SEQUENTIAL_FEATURE_EXTRACTION = false;
		NetworkConfig._CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.0;
		NetworkConfig._numThreads = 8;
		
		NetworkConfig.USE_STRUCTURED_SVM = true;

		// Set weight to not random to make useful comparison between sequential and parallel touch
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig.FEATURE_INIT_WEIGHT = 0.0;
		
		int numIterations = 200;
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		LinearCRFFeatureManager fm = new LinearCRFFeatureManager(new GlobalNetworkParam());
		
		LinearCRFNetworkCompiler compiler = new LinearCRFNetworkCompiler();
		
		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);
		
		model.train(trainInstances, numIterations);
		
		Instance[] predictions = model.decode(testInstances);
		
		int corr = 0;
		int total = 0;
		int count = 0;
		for(Instance ins: predictions){
			LinearCRFInstance instance = (LinearCRFInstance)ins;
			ArrayList<Label> goldLabel = instance.getOutput();
			ArrayList<Label> actualLabel = instance.getPrediction();
			ArrayList<String[]> words = instance.getInput();
			for(int i=0; i<goldLabel.size(); i++){
				if(goldLabel.get(i).equals(actualLabel.get(i))){
					corr++;
				}
				total++;
				if(count < 3){
//					System.out.println(words.get(i)[0]+" "+words.get(i)[1]+" "+goldLabel.get(i).getId()+" "+actualLabel.get(i).getId());
					System.out.println(words.get(i)[0]+" "+goldLabel.get(i)+" "+actualLabel.get(i));
				}
			}
			count++;
			if(count < 3){
				System.out.println();
			}
		}
		System.out.println(String.format("Correct/Total: %d/%d", corr, total));
		System.out.println(String.format("Accuracy: %.2f%%", 100.0*corr/total));
	}
	
	private static LinearCRFInstance[] readCoNLLData(String fileName, boolean withLabels, boolean isLabeled, int number) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
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
	
	private static LinearCRFInstance[]  readCoNLLData(String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		return readCoNLLData(fileName, withLabels, isLabeled, -1);
	}
}
