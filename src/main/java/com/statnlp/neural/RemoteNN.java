package com.statnlp.neural;

import java.io.IOException;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;
import org.msgpack.core.MessageBufferPacker;
import org.msgpack.core.MessagePack;
import org.msgpack.core.MessageUnpacker;
import org.zeromq.ZMQ;

import com.statnlp.hybridnetworks.NetworkConfig;

public class RemoteNN {
	private boolean DEBUG = false;
	
	// Torch NN server information
	private ZMQ.Context context;
	private ZMQ.Socket requester;
	private String serverAddress = NeuralConfig.NEURAL_SERVER_PREFIX + NeuralConfig.NEURAL_SERVER_ADDRESS+":" + NeuralConfig.NEURAL_SERVER_PORT;
	
	// Reference to controller instance for updating weights and getting gradients
	private NNCRFInterface controller;
	
	public RemoteNN() {
		context = ZMQ.context(1);
		requester = context.socket(ZMQ.REQ);
		requester.connect(serverAddress);
	}
	
	public void setController(NNCRFInterface controller) {
		this.controller = controller;
	}
	
	public double[] initNetwork(List<Integer> numInputList, List<Integer> inputDimList,
						   List<String> embeddingList, List<Integer> embSizeList,
						   int outputDim, List<List<Integer>> vocab) {
		JSONObject obj = new JSONObject();
		obj.put("cmd", "init");
		obj.put("numInputList", numInputList);
		obj.put("inputDimList", inputDimList);
		obj.put("embedding", embeddingList);
		obj.put("embSizeList", embSizeList);
		obj.put("outputDim", outputDim);
		obj.put("numLayer", NeuralConfig.NUM_LAYER);
		obj.put("hiddenSize", NeuralConfig.HIDDEN_SIZE);
		obj.put("activation", NeuralConfig.ACTIVATION);
		obj.put("dropout", NeuralConfig.DROPOUT);
		obj.put("optimizer", NeuralConfig.OPTIMIZER);
		obj.put("learningRate", NeuralConfig.LEARNING_RATE);
		obj.put("vocab", vocab);

		String request = obj.toString();
		requester.send(request.getBytes(), 0);
		byte[] reply = requester.recv(0);
		double[] nnInternalWeights = null;
		if(NetworkConfig.OPTIMIZE_NEURAL) {
			JSONArray arr = new JSONArray(new String(reply));
			nnInternalWeights = new double[arr.length()];
			for (int i = 0; i < nnInternalWeights.length; i++) {
				nnInternalWeights[i] = arr.getDouble(i);
			}
		}
		if (DEBUG) {
			System.out.println("Init returns " + new String(reply));
		}
		return nnInternalWeights;
	}
	
	public void forwardNetwork(boolean training) {
		MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
		int mapSize = NetworkConfig.OPTIMIZE_NEURAL? 3:2;
		try {
			packer.packMapHeader(mapSize);
			packer.packString("cmd").packString("fwd");
			packer.packString("training").packBoolean(training);
			
			if(NetworkConfig.OPTIMIZE_NEURAL) {
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
				nnExternalWeights[i] = unpacker.unpackDouble();
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
		
		
	}
	
	public void backwardNetwork() {
		MessageBufferPacker packer = MessagePack.newDefaultBufferPacker();
		try {
			packer.packMapHeader(2);
			packer.packString("cmd").packString("fwd");
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
			if(NetworkConfig.OPTIMIZE_NEURAL) {
				int size = unpacker.unpackArrayHeader();
				double[] counts = new double[size];
				for (int i = 0; i < counts.length; i++) {
					counts[i] = unpacker.unpackDouble();
				}
				controller.setInternalNeuralGradients(counts);
			}
			if (DEBUG) {
				System.out.println("Backward returns " + new String(reply));
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
	}
	
	public void cleanUp() {
		requester.close();
		context.term();
	}
	
}
