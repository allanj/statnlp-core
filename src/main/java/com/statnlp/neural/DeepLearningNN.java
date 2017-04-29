package com.statnlp.neural;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer.Builder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * NN backend using DeepLearning4J
 */
public class DeepLearningNN {
	private boolean DEBUG = false;
	
	private static final int SEED = 1337;
	
	private ComputationGraph model;
	
	private Map<String,Integer> word2idx;
	private int numInput;
	private List<INDArray> x;
	
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
		this.x = prepareInput(vocab, numInputList, embSizeList);
		this.word2idx.clear();
		for (int i = 0; i < wordList.size(); i++) {
			this.word2idx.put(wordList.get(i), i);
		}
		this.numInput = vocab.size();
		
		GraphBuilder confBuilder = new NeuralNetConfiguration.Builder()
        	.seed(SEED)
        	.weightInit(WeightInit.XAVIER)
        	.updater(Updater.valueOf(NeuralConfig.OPTIMIZER.toUpperCase()))
        	.learningRate(NeuralConfig.LEARNING_RATE)
        	.graphBuilder();
        
		// build input layer
		String[] inputs = new String[inputDimList.size()];
		String[] inputLayers = new String[inputDimList.size()];
		for (int i = 0; i < inputDimList.size(); i++) {
			inputs[i] = "inputs"+i;
			inputLayers[i] = "I"+i;
		}
		confBuilder = confBuilder.addInputs(inputs);
		
		int totalDim = 0;
		for (int i = 0; i < inputDimList.size(); i++) {
			int inputDim = inputDimList.get(i);
			int embeddingSize = NeuralConfig.EMBEDDING_SIZE.get(i);
			if (embeddingSize == 0)
				embeddingSize = wordList.size();
			totalDim += numInputList.get(i) * embeddingSize;
			confBuilder = confBuilder.addLayer(inputLayers[i],
						new org.deeplearning4j.nn.conf.layers.EmbeddingLayer.Builder()
							.nIn(inputDim)
							.nOut(embeddingSize)
							.activation(Activation.IDENTITY)
							.biasInit(0.0)
							.biasLearningRate(0.0)
							.build(),
						inputs[i]);
		}

		// merge input embeddings 
		confBuilder = confBuilder.addVertex("merge", new MergeVertex(), inputLayers);
		
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
				confBuilder = confBuilder.addLayer(hiddenLayers[i],
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
			if (numLayer == 0) {
				lastInputDim = totalDim;
			} else {
				lastInputDim = hiddenSize;
			}
			Builder outputBuilder = new DenseLayer.Builder()
										.nIn(lastInputDim)
										.nOut(outputDim)
										.activation(Activation.IDENTITY);
			if (! NeuralConfig.USE_OUTPUT_BIAS) {
				outputBuilder = outputBuilder.biasInit(0.0).biasLearningRate(0.0);
			}
			DenseLayer outputLayer = outputBuilder.build();
			outputs[n] = "out"+n;
			confBuilder = confBuilder.addLayer(outputs[n], outputLayer, hiddenLayers[numLayer-1]);
		}
		
		ComputationGraphConfiguration conf = confBuilder.setOutputs(outputs)
											 .backprop(true).pretrain(false)
											 .build();
		ComputationGraph net = new ComputationGraph(conf);
		
		// TODO: initialize embedding layer
		
		net.init();
		net.params();
		
		return null;
		/*
		MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
		try {
			packer.packMapHeader(17);
			packer.packString("cmd").packString("init");
			
			packList(packer, "numInputList", numInputList);
			packList(packer, "inputDimList", inputDimList);
			packList(packer, "wordList", wordList);
			packer.packString("lang").packString(lang);
			packList(packer, "embedding", embeddingList);
			packList(packer, "embSizeList", embSizeList);
			packList(packer, "outputDimList", outputDimList);
//			packer.packString("outputDim").packInt(outputDimList.size());
			packer.packString("numLayer").packInt(NeuralConfig.NUM_LAYER);
			packer.packString("hiddenSize").packInt(NeuralConfig.HIDDEN_SIZE);
			packer.packString("activation").packString(NeuralConfig.ACTIVATION);
			packer.packString("dropout").packDouble(NeuralConfig.DROPOUT);
			packer.packString("optimizer").packString(NeuralConfig.OPTIMIZER);
			packer.packString("learningRate").packDouble(NeuralConfig.LEARNING_RATE);
			packer.packString("fixEmbedding").packBoolean(NeuralConfig.FIX_EMBEDDING);
			packer.packString("useOutputBias").packBoolean(NeuralConfig.USE_OUTPUT_BIAS);
			packList(packer, "vocab", vocab);
			packer.close();
			
			requester.send(packer.toByteArray(), 0);
			byte[] reply = requester.recv(0);
			double[] nnInternalWeights = null;
			if(optimizeNeural) {
				MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(reply);
				int size = unpacker.unpackArrayHeader();
				nnInternalWeights = new double[size];
				for (int i = 0; i < nnInternalWeights.length; i++) {
					nnInternalWeights[i] = unpackDoubleOrInt(unpacker);
				}
			}
			if (DEBUG) {
				System.out.println("Init returns " + new String(reply));
			}
			return nnInternalWeights;
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return null;
		*/
	}
	
	private List<INDArray> prepareInput(List<List<Integer>> vocab, List<Integer> numInputList, List<Integer> embSizeList) {
		List<INDArray> result = new ArrayList<INDArray>();
		int startIdx = 0;
		for (int i = 0; i < numInputList.size(); i++) {
			result.add(Nd4j.createUninitialized(new int[]{vocab.size(), numInputList.get(i)}));
			for (int j = 0; j < vocab.size(); j++) {
				for (int k = 0; k < numInputList.get(i); k++) {
					// result[i][j][k] = vocab[j][startIdx+k]
					result.get(i).putScalar(new int[]{j,k}, vocab.get(j).get(startIdx+k));
				}
			}
			startIdx = startIdx + numInputList.get(i);
		}
		return result;
	}
	
	public void forwardNetwork(boolean training) {
		/*
		MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
		int mapSize = optimizeNeural ? 3 : 2;
		try {
			packer.packMapHeader(mapSize);
			packer.packString("cmd").packString("fwd");
			packer.packString("training").packBoolean(training);
			
			if(optimizeNeural) {
				double[] nnInternalWeights = controller.getInternalNeuralWeights();
				packer.packString("weights");
				packer.packArrayHeader(nnInternalWeights.length);
				for (int i = 0; i < nnInternalWeights.length; i++) {
					packer.packDouble(nnInternalWeights[i]);
				}
			}
			packer.close();
			
			requester.send(packer.toByteArray(), 0);
			byte[] reply = requester.recv(0);
			
			MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(reply);
			int size = unpacker.unpackArrayHeader();
			double[] nnExternalWeights = new double[size];
			for (int i = 0; i < size; i++) {
				nnExternalWeights[i] = unpackDoubleOrInt(unpacker);
			}
			controller.updateExternalNeuralWeights(nnExternalWeights);
			unpacker.close();
			if (DEBUG) {
				System.out.println("Forward returns " + reply.toString());
			}
			
		} catch (IOException e) {
			System.err.println("Exception happened while forwarding network...");
			e.printStackTrace();
		}
		
		*/
	}
	
	public void backwardNetwork() {
		/*
		MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
		try {
			packer.packMapHeader(2);
			packer.packString("cmd").packString("bwd");
			double[] grad = controller.getExternalNeuralGradients();
			packer.packString("grad");
			packer.packArrayHeader(grad.length);
			for (int i = 0; i < grad.length; i++) {
				packer.packDouble(grad[i]);
			}
			packer.close();
			requester.send(packer.toByteArray(), 0);
			
			byte[] reply = requester.recv(0);
			MessageUnpacker unpacker = MessagePack.newDefaultUnpacker(reply);
			if(optimizeNeural) {
				int size = unpacker.unpackArrayHeader();
				double[] counts = new double[size];
				for (int i = 0; i < counts.length; i++) {
					counts[i] = unpackDoubleOrInt(unpacker);
				}
				controller.setInternalNeuralGradients(counts);
			}
			if (DEBUG) {
				System.out.println("Backward returns " + new String(reply));
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		*/
		
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
	
	public INDArray nonInputParams(ComputationGraph net, int numInput) {
		List<INDArray> list = new ArrayList<INDArray>();
		for(int i = numInput; i < net.getNumLayers(); i++) {
			Layer l = net.getLayer(i);
			INDArray layerParams = l.params();
			if (layerParams != null) {
				list.add(layerParams);
			}
        }
        return Nd4j.toFlattened('f', list);
    }
}
