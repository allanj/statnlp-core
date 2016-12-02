package com.statnlp.neural.test;

import static org.junit.Assert.assertEquals;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.statnlp.neural.model.MultiLayerPerceptron;
import org.statnlp.neural.module.Linear;
import org.statnlp.neural.module.LookupTable;
import org.statnlp.neural.module.ParallelTable;
import org.statnlp.neural.module.Sequential;
import org.statnlp.neural.util.INDArrayList;
import org.statnlp.neural.util.Config.Activation;
import org.statnlp.neural.util.Config.WordEmbedding;

public class NeuralTest {

	private static MultiLayerPerceptron mlp;
	
	public NeuralTest() {
	}

	public static void main(String[] args) {
		System.out.println("*** MultiLayerPerceptronTest ***");
		mlp = new MultiLayerPerceptron(1, 2, Activation.TANH, new int[]{4,2}, new int[]{2,1}, WordEmbedding.RANDOM, new int[]{2,1}, 1, true);
		
		Sequential seq = mlp.getModules();
		
		ParallelTable pt = (ParallelTable) seq.get(0);
		Sequential innerSeq1 = (Sequential) pt.get(0);
		LookupTable lt1 = (LookupTable) innerSeq1.get(0);
		double[][] lt1Weight = new double[][] {{0.206504, -0.261428}, {0.418374, 0.414750}, {-0.080426, -0.149622}, {-0.441742, 0.393428}};
		lt1.setWeight(Nd4j.create(lt1Weight));
		
		Sequential innerSeq2 = (Sequential) pt.get(1);
		LookupTable lt2 = (LookupTable) innerSeq2.get(0);
		double[][] lt2Weight = new double[][] {{0.5},{-0.2}};
		lt2.setWeight(Nd4j.create(lt2Weight));
		
		Linear lin1 = (Linear) seq.get(2);
		double[][] lin1Weight = new double[][] {
				{0.206504, -0.261428, 0.418374, 0.414750, 0.399983},
				{-0.080426, -0.149622, -0.441742, 0.393428, 0.390938}
		};
		lin1.setWeight(Nd4j.create(lin1Weight));
		lin1.setBias(Nd4j.create(new double[] {-0.091970, -0.136384}));
		
		Linear lin2 = (Linear) seq.get(4);
		double[][] lin2Weight = new double[][] {{0.123, -0.432}};
		lin2.setWeight(Nd4j.create(lin2Weight));
		lin2.setBias(Nd4j.create(new double[] {1.34}));
		System.out.println();
		testForwardBackword();
	}
	
	private static void testForwardBackword() {
		System.out.println("testForwardBackward()");
		INDArray inp1 = Nd4j.create(new double[][] {{0,1},{2,3}});
		INDArray inp2 = Nd4j.create(new double[][] {{1},{0}});
		INDArray input = new INDArrayList(inp1,inp2);
		INDArray output = mlp.forward(input);
		System.out.println("> output:\n" + output);
		double[][] expected = new double[][] {
				{1.4651976643963}, {1.1754127011778}
		};
		assertEquals(Nd4j.create(expected), output);
		
		INDArray gradOutput = Nd4j.create(new double[][] {{0.15}, {-0.3}});
		mlp.backward(input, gradOutput);
		
		Sequential seq = mlp.getModules();
		ParallelTable pt = (ParallelTable) seq.get(0);
		Sequential innerSeq1 = (Sequential) pt.get(0);
		LookupTable lt1 = (LookupTable) innerSeq1.get(0);
		System.out.println("> gradWeight of LT1:\n" + lt1.getGradWeight());
		expected = new double[][] {
				{0.0084951036093485, 0.0048166748595161}, 
				{0.034476614375003, -0.017305851440989}, 
				{-0.016185402389435, -0.0065683623350798}, 
				{-0.062794462037469, 0.027214521313061}
		};
		assertEquals(Nd4j.create(expected), lt1.getGradWeight());
		
		Sequential innerSeq2 = (Sequential) pt.get(1);
		LookupTable lt2 = (LookupTable) innerSeq2.get(0);
		System.out.println("> gradWeight of LT2:\n" + lt2.getGradWeight());
		expected = new double[][] {
				{0.02748505209594}, 
				{-0.017402965859778}
		};
		assertEquals(Nd4j.create(expected), lt2.getGradWeight());
		
		Linear lin1 = (Linear) seq.get(2);
		System.out.println("> gradInput of Lin1:\n" + lin1.getGradInput());
		expected = new double[][] {
				{0.0084951036093485, 0.0048166748595161, 0.034476614375003, -0.017305851440989, -0.017402965859778}, 
				{-0.016185402389435, -0.0065683623350798, -0.062794462037469, 0.027214521313061, 0.02748505209594}
		};
		assertEquals(Nd4j.create(expected), lin1.getGradInput());
		
		Linear lin2 = (Linear) seq.get(4);
		System.out.println("> gradInput of Lin2:\n" + lin2.getGradInput());
		expected = new double[][] {
				{0.01845, -0.0648}, 
				{-0.0369, 0.1296} 
		};
		assertEquals(Nd4j.create(expected), lin2.getGradInput());
	}
}
