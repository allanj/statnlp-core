package com.statnlp.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import com.statnlp.commons.types.Instance;
import com.statnlp.example.linear_ie.AttributedWord;
import com.statnlp.example.linear_ie.Label;
import com.statnlp.example.linear_ie.LinearIEFeatureManager;
import com.statnlp.example.linear_ie.LinearIEInstance;
import com.statnlp.example.linear_ie.LinearIENetworkCompiler;
import com.statnlp.example.linear_ie.Span;
import com.statnlp.example.linear_ie.LinearIEFeatureManager.FeatureType;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GenerativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;

public class LinearIEMain {
	
	public static ArrayList<Label> labels;
	
	public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException, IllegalArgumentException, IllegalAccessException, NoSuchFieldException, SecurityException{
		boolean serializeModel = true;
		
		String train_filename = "data/ACE2004/data/English/mention-standard/FINE_TYPE/train.data.500";
		String test_filename = "data/ACE2004/data/English/mention-standard/FINE_TYPE/test.data";
		
		LinearIEInstance[] trainInstances = readData(train_filename, true, true);
		LinearIEInstance[] testInstances = readData(test_filename, true, false);
		
		labels = new ArrayList<Label>();
		labels.addAll(Label.LABELS.values());
		int maxSize = 0;
		for(LinearIEInstance instance: trainInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		for(LinearIEInstance instance: testInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig._CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.01;
		NetworkConfig.objtol = 1e-4;
		NetworkConfig._numThreads = 4;
		
		int numIterations = 2500;
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		LinearIEFeatureManager fm = new LinearIEFeatureManager(new GlobalNetworkParam());
		
		LinearIENetworkCompiler compiler = new LinearIENetworkCompiler(labels.toArray(new Label[labels.size()]), maxSize);
		
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler) : DiscriminativeNetworkModel.create(fm, compiler);
		
		if(serializeModel){
			String modelPath = "experiments/mention/model/FINE_TYPE/ACE2004/English/aldrian.0.01.allfeatures.500data.withmp.optimal.model";
			if(new File(modelPath).exists()){
				System.out.println("Reading object...");
				long startTime = System.currentTimeMillis();
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
				model = (NetworkModel)ois.readObject();
				ois.close();
				Field _fm = NetworkModel.class.getDeclaredField("_fm");
				_fm.setAccessible(true);
				fm = (LinearIEFeatureManager)_fm.get(model);
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			} else {
				model.train(trainInstances, numIterations);
				System.out.println("Writing object...");
				long startTime = System.currentTimeMillis();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
				oos.writeObject(model);
				oos.close();
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			}
		} else {
			model.train(trainInstances, numIterations);
		}
		
		int mentionPenaltyFeatureIndex = fm.getParam_G().toFeature(FeatureType.MENTION_PENALTY.name(), "MP", "MP");
		
		for(double mentionPenalty = -0.2; mentionPenalty < 1.0; mentionPenalty += 0.2){
			if(mentionPenalty >= 0.0){
				fm.getParam_G().setWeight(mentionPenaltyFeatureIndex, mentionPenalty);
			}
			System.out.println(String.format("Mention penalty: %.1f", fm.getParam_G().getWeight(mentionPenaltyFeatureIndex)));
			Instance[] predictions = model.decode(testInstances);
			int corr = 0;
			int totalGold = 0;
			int totalPred = 0;
			for(Instance inst: predictions){
				LinearIEInstance instance = (LinearIEInstance)inst;
				System.out.println("Words:");
				System.out.println(toString(instance.words));
				System.out.println("Gold:");
				System.out.println(instance.output);
				System.out.println("Prediction:");
				System.out.println(instance.prediction);
				System.out.println();
				List<Span> goldSpans = instance.output;
				List<Span> predSpans = instance.prediction;
				int curTotalGold = goldSpans.size();
				totalGold += curTotalGold;
				int curTotalPred = predSpans.size();
				totalPred += curTotalPred;
				int curCorr = countOverlaps(goldSpans, predSpans);
				corr += curCorr;
	//			if(curTotalPred == 0) curTotalPred = 1;
	//			if(curTotalGold == 0) curTotalGold = 1;
				double precision = 100.0*curCorr/curTotalPred;
				double recall = 100.0*curCorr/curTotalGold;
				double f1 = 2/((1/precision)+(1/recall));
				if(curTotalPred == 0) precision = 0.0;
				if(curTotalGold == 0) recall = 0.0;
				if(curTotalPred == 0 || curTotalGold == 0) f1 = 0.0;
				System.out.println("Correct spans: "+curCorr);
				System.out.println("Gold spans: "+curTotalGold);
				System.out.println("Predicted spans: "+curTotalPred);
				System.out.println(String.format("P: %.2f%%", precision));
				System.out.println(String.format("R: %.2f%%", recall));
				System.out.println(String.format("F: %.2f%%", f1));
			}
			System.out.println("Correct spans: "+corr);
			System.out.println("Gold spans: "+totalGold);
			System.out.println("Predicted spans: "+totalPred);
	//		if(totalPred == 0) totalPred = 1;
	//		if(totalGold == 0) totalGold = 1;
			double precision = 100.0*corr/totalPred;
			double recall = 100.0*corr/totalGold;
			double f1 = 2/((1/precision)+(1/recall));
			if(totalPred == 0) precision = 0.0;
			if(totalGold == 0) recall = 0.0;
			if(totalPred == 0 || totalGold == 0) f1 = 0.0;
			System.out.println(String.format("P: %.2f%%", precision));
			System.out.println(String.format("R: %.2f%%", recall));
			System.out.println(String.format("F: %.2f%%", f1));
		}
	}

	private static String toString(Object[] arr){
		StringBuilder builder = new StringBuilder();
//		builder.append("[");
		int index = 0;
		for(Object str: arr){
			if(builder.length() > 0) builder.append(" ");
			builder.append(str+"("+index+")");
			index++;
		}
//		builder.append("]");
		return builder.toString();
	}
	
	/**
	 * Count the number of overlaps (common elements) in the given lists.
	 * Duplicate objects are counted as separate objects.
	 * @param list1
	 * @param list2
	 * @return
	 */
	private static int countOverlaps(List<Span> list1, List<Span> list2){
		int result = 0;
		List<Span> copy = new ArrayList<Span>();
		copy.addAll(list2);
		for(Span span: list1){
			if(copy.contains(span)){
				copy.remove(span);
				result += 1;
			}
		}
		return result;
	}
	
	/**
	 * Read a list of instances from a file
	 * @param fileName
	 * @param withLabels
	 * @param isLabeled
	 * @return
	 * @throws IOException
	 */
	private static LinearIEInstance[] readData(String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<LinearIEInstance> result = new ArrayList<LinearIEInstance>();
		int instanceId = 1;
		while(br.ready()){
			String words = br.readLine();
			LinearIEInstance instance = new LinearIEInstance(instanceId++, 1.0);
			instance.words = markWords(words.trim().split(" "));
			String posTags = br.readLine();
			instance.posTags = posTags.trim().split(" ");
			String[] spans = br.readLine().split("\\|");
			if(spans.length == 1 && spans[0].length() == 0){
				spans = new String[0];
			}
			List<Span> output = new ArrayList<Span>();
			for(String span: spans){
				String[] tokens = span.split(" ");
				String[] indices = tokens[0].split(",");
				int[] intIndices = new int[indices.length];
				for(int i=0; i<4; i++){
					intIndices[i] = Integer.parseInt(indices[i]);
				}
				Label label = Label.get(tokens[1]);
				output.add(new Span(intIndices[0], intIndices[1], intIndices[2], intIndices[3], label));
			}
			instance.setOutput(output);
			if(isLabeled){
				instance.setLabeled();
			} else {
				instance.setUnlabeled();
			}
			br.readLine();
			result.add(instance);
		}
		br.close();
		return result.toArray(new LinearIEInstance[result.size()]);
	}
	
	private static AttributedWord[] markWords(String[] words){
		AttributedWord[] result = new AttributedWord[words.length];
		for(int i=0; i<result.length; i++){
			result[i] = new AttributedWord(words[i]);
		}
		return result;
	}
}