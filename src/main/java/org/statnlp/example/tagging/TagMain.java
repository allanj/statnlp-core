package org.statnlp.example.tagging;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.statnlp.commons.io.RAWF;
import org.statnlp.commons.ml.opt.GradientDescentOptimizer.BestParamCriteria;
import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.decoding.Metric;
import org.statnlp.hypergraph.neural.BidirectionalLSTM;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class TagMain {

	public static String trainFile = "data/conll2000/sample_train.txt";
	public static String testFile = "data/conll2000/sample_train.txt";
	public static int trainNum = 30;
	public static int testNum = 1;
	public static int numThreads = 1;
	public static double l2 = 0.01;
	public static int numIterations = 1000;
	public static List<String> labels;
	public static boolean visualization = false;
	public static int maxLen;
	
	public static Map<String, Integer> word2int = new HashMap<>();
	
	public static void main(String[] args) throws IOException, InterruptedException{
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.USE_NEURAL_FEATURES = true;
		NetworkConfig.AVOID_DUPLICATE_FEATURES = true;

		//If you want to run Neural-CRF on Linux, please uncomment this line.
		//NetworkConfig.OS = "linux";

		labels = new ArrayList<>();
		word2int.put("<pad>", 1);
		word2int.put("<unk>", 1);
		TagInstance[] trainInstances = readData(trainFile, true, trainNum);
		TagInstance[] devInstances = readData(trainFile, false, 2);
		System.out.println("#labels: " + labels.size());
		NetworkConfig.MODEL_TYPE = ModelType.CRF;
		
		List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
		OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
		BidirectionalLSTM net = null;
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			NetworkConfig.USE_BATCH_TRAINING = true;
			NetworkConfig.BATCH_SIZE = 10;
			NetworkConfig.FEATURE_TOUCH_TEST = true;
			NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.01;
			net = new BidirectionalLSTM(50, labels.size(), word2int);
			net.setMaxLen(maxLen);
			optimizer = OptimizerFactory.getGradientDescentFactory(BestParamCriteria.BEST_ON_DEV, 0.015);
			nets.add(net);
		}
		GlobalNeuralNetworkParam gnnp = new GlobalNeuralNetworkParam(nets);
		
		GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, gnnp);
		TagFeatureManager fa = new TagFeatureManager(gnp);
		TagNetworkCompiler compiler = new TagNetworkCompiler(labels);
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		
		Function<Instance[], Metric> evalFunc = new Function<Instance[], Metric>() {

			@Override
			public Metric apply(Instance[] t) {
				int corr = 0;
				int total = 0;
				for (Instance pred : t) {
					TagInstance inst = (TagInstance)pred;
					List<String> gold = inst.getOutput();
					List<String> prediction = inst.getPrediction();
					for (int i = 0; i < gold.size(); i++) {
						if (gold.get(i).equals(prediction.get(i)))
							corr++;
					}
					total += gold.size();
				}
				System.out.printf("[Accuracy]: %.2f%%\n", corr * 1.0 / total * 100);
				return new AccMetric(corr * 1.0 / total * 100);
			}
			
		};
		
		model.train(trainInstances, numIterations, devInstances, evalFunc, 30);
		if (visualization) model.visualize(TaggingViewer.class, trainInstances);
		
		TagInstance[] testInstances = readData(testFile, false, testNum);
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			net.setMaxLen(maxLen);
		}
		Instance[] predictions = model.test(testInstances);
		
		//evaluation
		int corr = 0;
		int total = 0;
		for (Instance pred : predictions) {
			TagInstance inst = (TagInstance)pred;
			List<String> gold = inst.getOutput();
			List<String> prediction = inst.getPrediction();
			for (int i = 0; i < gold.size(); i++) {
				if (gold.get(i).equals(prediction.get(i)))
					corr++;
			}
			total += gold.size();
		}
		System.out.printf("[Accuracy]: %.2f%%\n", corr * 1.0 / total * 100);
	}
	
	/**
	 * Read the data.
	 * @param path
	 * @param isTraining
	 * @param number
	 * @return
	 * @throws IOException
	 */
	public static TagInstance[] readData(String path, boolean isTraining, int number) throws IOException{
		BufferedReader br = RAWF.reader(path);
		String line = null;
		List<TagInstance> insts = new ArrayList<TagInstance>();
		int index =1;
		ArrayList<WordToken> words = new ArrayList<WordToken>();
		ArrayList<String> tags = new ArrayList<String>();
		while((line = br.readLine())!=null){
			if(line.equals("")){
				WordToken[] wordsArr = new WordToken[words.size()];
				words.toArray(wordsArr);
				maxLen = Math.max(maxLen, words.size());
				Sentence sent = new Sentence(wordsArr);
				TagInstance inst = new TagInstance(index++, 1.0, sent, tags);
				if(isTraining) {
					inst.setLabeled(); 
				} else {
					inst.setUnlabeled();
				}
				insts.add(inst);
				words = new ArrayList<WordToken>();
				tags = new ArrayList<String>();
				if(number!=-1 && insts.size()==number) break;
				continue;
			}
			String[] values = line.split(" ");
			String word = values[0];
			String tag = values[1];
			if (isTraining && !labels.contains(tag))
				labels.add(tag);
			if (isTraining) {
				if (!word2int.containsKey(word)) {
					word2int.put(word, word2int.size());
				}
			}
			words.add(new WordToken(word));
			tags.add(tag);
		}
		br.close();
		List<TagInstance> myInsts = insts;
		System.out.println("#instance:"+ myInsts.size()+" Instance. ");
		return myInsts.toArray(new TagInstance[myInsts.size()]);
	}


}
