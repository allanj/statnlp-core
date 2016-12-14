package com.statnlp.neural;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

import org.statnlp.neural.util.Config.Activation;
import org.statnlp.neural.util.Config.WordEmbedding;

public class NeuralConfigReader {
	public static void readConfig(String filename) throws FileNotFoundException {
		Scanner scan = new Scanner(new File(filename));
		while(scan.hasNextLine()){
			String line = scan.nextLine().trim();
			if(line.equals("")){
				continue;
			}
			String[] info = line.split(" ");
			if(info[0].equals("serverAddress")) {
				NeuralConfig.NEURAL_SERVER_ADDRESS = info[1];
			} else if(info[0].equals("serverPort")) {
				NeuralConfig.NEURAL_SERVER_PORT= Integer.parseInt(info[1]);
			} else if(info[0].equals("lang")) {
				NeuralConfig.LANGUAGE = info[1];
			} else if(info[0].equals("wordEmbedding")) {  //senna glove polygot
				NeuralConfig.EMBEDDING = new ArrayList<WordEmbedding>();
				for (int i = 1; i < info.length; i++) {
					NeuralConfig.EMBEDDING.add(WordEmbedding.valueOf(info[i].toUpperCase()));
				}
			} else if(info[0].equals("embeddingPath")) {
				NeuralConfig.EMBEDDING_PATH = info[1];
			} else if(info[0].equals("embeddingSize")) {
				NeuralConfig.EMBEDDING_SIZE = new ArrayList<Integer>();
				for (int i = 1; i < info.length; i++) {
					NeuralConfig.EMBEDDING_SIZE.add(Integer.parseInt(info[i])); 
				}
			} else if(info[0].equals("numLayer")) {
				NeuralConfig.NUM_LAYER = Integer.parseInt(info[1]);
			} else if(info[0].equals("hiddenSize")) {
				NeuralConfig.HIDDEN_SIZE = Integer.parseInt(info[1]);
			} else if(info[0].equals("activation")) { //tanh, relu, identity, hardtanh
				NeuralConfig.ACTIVATION = Activation.valueOf(info[1].toUpperCase());
			} else if(info[0].equals("dropout")) {
				NeuralConfig.DROPOUT = Double.parseDouble(info[1]);
			} else if(info[0].equals("optimizer")) {  //adagrad, adam, sgd , none(be careful with the config in statnlp)
				NeuralConfig.OPTIMIZER = info[1];
			} else if(info[0].equals("learningRate")) {
				NeuralConfig.LEARNING_RATE = Double.parseDouble(info[1]);
			} else if(info[0].equals("fixEmbedding")) {
				NeuralConfig.FIX_EMBEDDING = true;
			} else {
				System.err.println("Unrecognized option: " + line);
			}
		}
		
		scan.close();
	}
}
