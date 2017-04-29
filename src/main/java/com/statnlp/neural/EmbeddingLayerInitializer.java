package com.statnlp.neural;

import java.util.Arrays;

import org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EmbeddingLayerInitializer {
	public static enum Embedding {
		ONEHOT, POLYGLOT
	};
	
	public static void initializeEmbeddingLayer(EmbeddingLayer layer, String embeddingName, int vocabSize, int embeddingSize) {
		Embedding embedding = Embedding.valueOf(embeddingName);
		switch (embedding) {
		case ONEHOT:
			oneHot(layer, vocabSize, embeddingSize); break;
		}
		// default does nothing
	}
	
	public static void oneHot(EmbeddingLayer layer, int vocabSize, int embeddingSize) {
		double[][] initWeights = new double[vocabSize][vocabSize];
		Arrays.fill(initWeights, 0.0);
		for (int i = 0; i < vocabSize; i++) {
			initWeights[i][i] = 1.0;
		}
		setWeights(layer, initWeights);
	}
	
	private static void setWeights(EmbeddingLayer layer, double[][] initWeights) {
		String key = DefaultParamInitializer.WEIGHT_KEY;
		INDArray weights = layer.getParam(key);
		// putting pre-trained weights into rows
		int vocabSize = weights.size(0);
		int embeddingSize = weights.size(1);
		INDArray rows = Nd4j.createUninitialized(new int[]{vocabSize, embeddingSize}, 'c');
		for (int i = 0; i < vocabSize; i++) {
		    double[] embeddings = initWeights[i];
		    INDArray newArray = Nd4j.create(embeddings);
		    rows.putRow(i, newArray);
		}

		// finally put rows in place of weights
		layer.setParam(key, rows);
	}
}
