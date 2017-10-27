package org.statnlp.example.benchmark_pytorch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.ml.opt.GradientDescentOptimizer.BestParamCriteria;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;

public class BMMain {

	public static int trainNum = 30;
	public static int testNum = 20;
	public static int numThreads = 1;
	public static double l2 = 0;
	public static int numIterations = 200;
	public static List<String> labels;
	public static Map<String, Integer> labelMap;
	public static int hiddenSize = 2;
	public static int embeddingSize = 5;
	
	public static void main(String[] args) throws IOException, InterruptedException{
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		NetworkConfig.USE_NEURAL_FEATURES = true;

		//If you want to run Neural-CRF on Linux, please uncomment this line.
		NetworkConfig.OS = "osx";

		//following the order of PyTorch CRF
		labels = new ArrayList<>();
		labels.add("B");  
		labels.add("I");
		labels.add("O");
		labels.add(BMConfig.START);
		labels.add(BMConfig.END);
		labelMap = new HashMap<>();
		for (int i = 0; i < labels.size(); i++) {
			labelMap.put(labels.get(i), i);
		}
		
		BMInstance[] trainInstances = getExampleTrainData();
		System.out.println(labels.toString());
		System.out.println("#labels: " + labels.size());
		NetworkConfig.MODEL_TYPE = ModelType.CRF;
		
		NetworkConfig.RANDOM_INIT_WEIGHT = false;
		NetworkConfig.FEATURE_INIT_WEIGHT = 2;
		NetworkConfig.USE_BATCH_TRAINING = true;
		NetworkConfig.BATCH_SIZE = 1;
		NetworkConfig.PRINT_BATCH_OBJECTIVE = true;
		
		List<NeuralNetworkCore> nets = new ArrayList<NeuralNetworkCore>();
		if (NetworkConfig.USE_NEURAL_FEATURES) {
			BMBiLSTM net = new BMBiLSTM(labels.size(), hiddenSize, embeddingSize);
			nets.add(net);
		}
		GlobalNeuralNetworkParam gnnp = new GlobalNeuralNetworkParam(nets);
		GlobalNetworkParam gnp = new GlobalNetworkParam(OptimizerFactory.getGradientDescentFactory(0.01), gnnp);
		BMFeatureManager fa = new BMFeatureManager(gnp, labels);
		BMNetworkCompiler compiler = new BMNetworkCompiler(labels);
		NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
		model.train(trainInstances, numIterations);
		
	}
	
	
	public static BMInstance[] getExampleTrainData() {
		String[] inputs = new String[]{"the wall street journal reported today that apple corporation made money" 
				,"georgia tech is a university in georgia python tensoflow pytorch torch "};
		String[] outputs = new String[]{"B I I I O O O B I O O", "B I O O O O B O B B B"};
		BMInstance[] insts = new BMInstance[inputs.length];
		for (int d = 0; d < inputs.length; d++) {
			String input = inputs[d];
			String output = outputs[d];
			String[] vals = input.split(" ");
			WordToken[] wordsArr = new WordToken[vals.length];
			List<String> outputLabels = new ArrayList<>(vals.length);
			String[] outputVals = output.split(" ");
			for(int w = 0; w < wordsArr.length; w++) {
				wordsArr[w] = new WordToken(vals[w]);
				outputLabels.add(outputVals[w]);
			}
			Sentence sent = new Sentence(wordsArr);
			insts[d] = new BMInstance(d + 1, 1.0, sent, outputLabels);
			insts[d].setLabeled();
		}
		return insts;
	}
	


}
