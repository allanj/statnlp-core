package com.statnlp.neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer.Builder;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.statnlp.hybridnetworks.NetworkConfig;

/**
 * NN backend using DeepLearning4J
 */
public class DeepLearningNN extends AbstractNN {
	private boolean DEBUG = true;
	
	private ComputationGraph model;
	private ComputationGraph inputLayer;
	private INDArray[] grads;
	
	private Map<String,Integer> word2idx;
	private int numInput;
	private INDArray[] x;
	private List<Integer> outputDimList;
	private int numOutput;
	
	// Reference to controller instance for updating weights and getting gradients
	private NNCRFInterface controller;
	
	// whether to use CRF's optimizer to optimize internal neural parameters
	private boolean optimizeNeural;
	
	public DeepLearningNN() {
		this(false);
	}
	
	public DeepLearningNN(boolean optimizeNeural) {
		this.optimizeNeural = optimizeNeural;
		this.word2idx = new HashMap<String, Integer>();
	}
	
	public void setController(NNCRFInterface controller) {
		this.controller = controller;
	}
	
	
	public double[] initNetwork(List<Integer> numInputList, List<Integer> inputDimList, List<String> wordList,
						   String lang, List<String> embeddingList, List<Integer> embSizeList,
						   List<Integer> outputDimList, List<List<Integer>> vocab) {
		int seed = NetworkConfig.RANDOM_INIT_FEATURE_SEED;
		
		this.x = prepareInput(vocab, numInputList, embSizeList, inputDimList);
		this.word2idx.clear();
		for (int i = 0; i < wordList.size(); i++) {
			this.word2idx.put(wordList.get(i), i);
		}
		this.numInput = vocab.size();
		this.outputDimList = outputDimList;
		this.numOutput = outputDimList.size();
		
		boolean fixEmbedding = NeuralConfig.FIX_EMBEDDING;
		GraphBuilder inputLayerBuilder = null;
		
		WeightInit weightInit;
		if (DEBUG) {
			weightInit = WeightInit.ZERO;
		} else {
			weightInit = WeightInit.XAVIER;
		}
		
		if (fixEmbedding) {
			inputLayerBuilder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.weightInit(weightInit)
				.updater(Updater.NONE)
				.learningRate(0.0)
				.graphBuilder();
		}
		
		Updater updater = Updater.valueOf(NeuralConfig.OPTIMIZER.toUpperCase());
		double learningRate = NeuralConfig.LEARNING_RATE;
		if (updater == Updater.NONE) {
			updater = Updater.SGD;
			learningRate = 1.0;
		}
		
		GraphBuilder mlpBuilder = new NeuralNetConfiguration.Builder()
        	.seed(seed)
        	.weightInit(weightInit)
        	.updater(updater)
        	.learningRate(learningRate)
        	.graphBuilder();
        
		// build input layer
		String[][] inputs = new String[numInputList.size()][];
		String[] stacks = new String[numInputList.size()];
		String[] inputLayers = new String[numInputList.size()];
		int numInputs = 0;
		for (int i = 0; i < numInputList.size(); i++) {
			inputs[i] = new String[numInputList.get(i)];
			for (int j = 0; j < numInputList.get(i); j++) {
				inputs[i][j] = "inputs"+i+"-"+j;
				numInputs++;
			}
			stacks[i] = "stacks"+i;
			inputLayers[i] = "I"+i;
		}
		
		String[] flatInputs = new String[numInputs];
		int ptr = 0;
		for (int i = 0; i < numInputList.size(); i++) {
			for (int j = 0; j < numInputList.get(i); j++) {
				flatInputs[ptr++] = inputs[i][j];
			}
		}
		String[] unstacks = new String[numInputs];
		
		if (fixEmbedding) {
			inputLayerBuilder.addInputs(flatInputs);
			mlpBuilder.addInputs("merge");
		} else {
			mlpBuilder.addInputs(flatInputs);
		}
		
		GraphBuilder whichBuilder = fixEmbedding ? inputLayerBuilder : mlpBuilder;
		for (int i = 0; i < numInputList.size(); i++) {
			int embeddingSize = NeuralConfig.EMBEDDING_SIZE.get(i);
			if (embeddingSize > 0) {
				whichBuilder.addVertex(stacks[i], new StackVertex(), inputs[i]);
			}
		}
		
		int totalDim = 0;
		for (int i = 0; i < inputDimList.size(); i++) {
			int inputDim = inputDimList.get(i);
			int embeddingSize = NeuralConfig.EMBEDDING_SIZE.get(i);
			if (embeddingSize > 0) {
				whichBuilder.addLayer(inputLayers[i],
						new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
							.nIn(inputDim)
							.nOut(embeddingSize)
							.activation(Activation.IDENTITY)
							.biasInit(0.0)
							.biasLearningRate(0.0)
							.build(),
						stacks[i]);
			} else {
				embeddingSize = inputDim;
			}
			totalDim += numInputList.get(i) * embeddingSize;
		}

		// merge input embeddings 
		ptr = 0;
		for (int i = 0; i < numInputList.size(); i++) {
			int stackSize = numInputList.get(i);
			int embeddingSize = NeuralConfig.EMBEDDING_SIZE.get(i);
			for (int j = 0; j < stackSize; j++) {
				if (embeddingSize > 0) {
					unstacks[ptr] = "unstack"+i+"-"+j;
					whichBuilder.addVertex(unstacks[ptr], new UnstackVertex(j, stackSize), inputLayers[i]);
				} else {
					unstacks[ptr] = "inputs"+i+"-"+j;
				}
				ptr++;
			}
		}
		
		whichBuilder.addVertex("merge", new MergeVertex(), unstacks);
		
		int numNetworks = outputDimList.size();
		int numLayer = NeuralConfig.NUM_LAYER;
		int hiddenSize = NeuralConfig.HIDDEN_SIZE;
		Activation activation = Activation.valueOf(NeuralConfig.ACTIVATION.toUpperCase());
		String[] outputs = new String[numNetworks];
		for (int n = 0; n < numNetworks; n++) {
			// build hidden layers
			String[] hiddenLayers = new String[numLayer];
			for (int i = 0; i < numLayer; i++) {
				hiddenLayers[i] = "H"+i+"N"+n;
			}
			for (int i = 0; i < numLayer; i++) {
				int denseInputSize;
				String denseInputName;
				if (i == 0) {
					denseInputSize = totalDim;
					denseInputName = "merge";
				} else {
					denseInputSize = hiddenSize;
					denseInputName = hiddenLayers[i-1];
				}
				mlpBuilder.addLayer(hiddenLayers[i],
						new DenseLayer.Builder()
							.nIn(denseInputSize)
							.nOut(hiddenSize)
							.activation(activation)
							.build(),
						denseInputName);
			}
			
			// build output layer
			int outputDim = outputDimList.get(n);
			int lastInputDim;
			String lastLayer;
			if (numLayer == 0) {
				lastInputDim = totalDim;
				lastLayer = "merge";
			} else {
				lastInputDim = hiddenSize;
				lastLayer = hiddenLayers[numLayer-1];
			}
			Builder outputBuilder = new DenseLayer.Builder()
										.nIn(lastInputDim)
										.nOut(outputDim)
										.activation(Activation.IDENTITY);
			if (! NeuralConfig.USE_OUTPUT_BIAS) {
				outputBuilder.biasInit(0.0).biasLearningRate(0.0);
			}
			DenseLayer outputLayer = outputBuilder.build();
			outputs[n] = "out"+n;
			mlpBuilder.addLayer(outputs[n], outputLayer, lastLayer);
		}
		
		if (fixEmbedding) {
			ComputationGraphConfiguration inputConf
				= inputLayerBuilder.setOutputs("merge").build();
			this.inputLayer = new ComputationGraph(inputConf);
		}
		ComputationGraphConfiguration mlpConf = mlpBuilder.setOutputs(outputs)
											 .backprop(true).pretrain(false)
											 .build();
		this.model = new ComputationGraph(mlpConf);
		this.model.init();
		System.out.println(mlpConf);
		
		// initialize embedding layer
		for (int i = 0; i < inputDimList.size(); i++) {
			int inputDim = inputDimList.get(i);
			String embeddingName = NeuralConfig.EMBEDDING.get(i);
			int embeddingSize = NeuralConfig.EMBEDDING_SIZE.get(i);
			if (embeddingSize > 0) {
				org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer layer;
				if (fixEmbedding) {
					layer = (org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer) this.inputLayer.getLayer(i);
				} else {
					layer = (org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer) this.model.getLayer(i);
				}
				EmbeddingLayerInitializer.initializeEmbeddingLayer(layer, embeddingName, inputDim, embeddingSize);
			}
		}
		
		System.out.println("Init DeepLearningNN");
		printParamsInfo();
		
        if (optimizeNeural) {
			return this.model.params().data().asDouble();
		} else {
			return null;
		}
	}
	
	private INDArray[] prepareInput(List<List<Integer>> vocab, List<Integer> numInputList, List<Integer> embSizeList, List<Integer> inputDimList) {
		List<INDArray> result = new ArrayList<INDArray>();
		int startIdx = 0;
		for (int i = 0; i < numInputList.size(); i++) {
//			result.add(Nd4j.createUninitialized(new int[]{vocab.size(), numInputList.get(i)}));
			for (int k = 0; k < numInputList.get(i); k++) {
				result.add(Nd4j.zeros(new int[]{vocab.size(), inputDimList.get(i)}));
			}
			for (int j = 0; j < vocab.size(); j++) {
				for (int k = 0; k < numInputList.get(i); k++) {
					// result[i][j][k] = vocab[j][startIdx+k]
					result.get(startIdx+k).putScalar(new int[]{j,vocab.get(j).get(startIdx+k)}, 1.0);
				}
			}
			startIdx = startIdx + numInputList.get(i);
		}
		INDArray[] arr = new INDArray[result.size()];
		return result.toArray(arr);
	}
	
	public void forwardNetwork(boolean training) {
		if (optimizeNeural) {
			double[] nnInternalWeights = controller.getInternalNeuralWeights();
			this.model.params().assign(Nd4j.create(nnInternalWeights));
		}

		INDArray[] _x = this.x;
	    if (NeuralConfig.FIX_EMBEDDING) {
	    	_x = this.inputLayer.output(this.x);
	    }
	    INDArray[] output = this.model.output(_x);
	    
	    int size = 0;
	    for (int i = 0; i < output.length; i++) {
	    	size += output[i].length();
	    }
	    double[] nnExternalWeights = new double[size];
	    int ptr = 0;
		for (int i = 0; i < output.length; i++) {
			for (int j = 0; j < output[i].rows(); j++) {
				for (int k = 0; k < output[i].columns(); k++) {
					nnExternalWeights[ptr++] = output[i].getDouble(j,k);
				}
			}
		}
		controller.updateExternalNeuralWeights(nnExternalWeights);
	}
	
	public void backwardNetwork() {
		double[] _grad = controller.getExternalNeuralGradients();
		if (this.grads == null) {
			this.grads = new INDArray[this.numOutput];
			for (int i = 0; i < this.numOutput; i++) {
				this.grads[i] = Nd4j.zeros(this.numInput, this.outputDimList.get(i));
			}
		}
		int ptr = 0;
		for (int i = 0; i < this.numOutput; i++) {
			for (int j = 0; j < this.numInput; j++) {
				for (int k = 0; k < this.outputDimList.get(i); k++) {
					this.grads[i].putScalar(new int[]{j,k}, _grad[ptr++]);
				}
			}
		}
		
		Gradient gradient = this.model.backpropGradient(this.grads);
		this.model.getUpdater().update(this.model, gradient, 0, 1);
		
		INDArray updateVector = gradient.gradient();
		if(optimizeNeural) {
			int size = updateVector.length();
			double[] counts = new double[size];
			for (int i = 0; i < counts.length; i++) {
				counts[i] = updateVector.getDouble(i);
			}
			controller.setInternalNeuralGradients(counts);
		} else {
	        this.model.params().subi(updateVector);
		}
	}
	
	public void saveNetwork(String prefix) {
		/*
		try {
			MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
			packer.packMapHeader(2);
			packer.packString("cmd").packString("save");
			packer.packString("savePrefix").packString(prefix);
			packer.close();
			requester.send(packer.toByteArray(), 0);
			
			byte[] reply = requester.recv(0);
			if (DEBUG) {
				System.out.println("Save returns " + new String(reply));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		*/
	}
	
	public void loadNetwork(String prefix) {
		/*
		try {
			MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
			packer.packMapHeader(2);
			packer.packString("cmd").packString("load");
			packer.packString("savePrefix").packString(prefix);
			packer.close();
			requester.send(packer.toByteArray(), 0);
			
			byte[] reply = requester.recv(0);
			if (DEBUG) {
				System.out.println("Save returns " + new String(reply));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		*/
	}

	@Override
	public void cleanUp() {
		
	}
	
	private void printParamsInfo() {
		//Print the  number of parameters in the network (and for each layer)
	    int totalNumParams = 0;
	    for( int i=0; i<model.getNumLayers(); i++ ){
	        int nParams = model.getLayer(i).numParams();
	        System.out.println("Number of parameters in layer " + i + ": " + nParams);
	        totalNumParams += nParams;
	    }
	    System.out.println("Total number of network parameters: " + totalNumParams);
	}
}
