package com.statnlp.neural;

import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EmbeddingLayerInitializer {
	public static enum Embedding {
		POLYGLOT, NONE
	};
	
	public static void initializeEmbeddingLayer(DenseLayer layer, String embeddingName, int vocabSize, int embeddingSize) {
		Embedding embedding = Embedding.valueOf(embeddingName.toUpperCase());
		/* TODO
		switch (embedding) {
		case POLYGLOT:
			polyglot(layer, vocabSize, embeddingSize); break;
		}
		*/
		// default does nothing
	}
	
	private static void setWeights(DenseLayer layer, double[][] initWeights) {
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
