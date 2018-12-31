package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.concatenate;
import static edu.cmu.dynet.internal.dynet_swig.exprPlus;
import static edu.cmu.dynet.internal.dynet_swig.exprTimes;
import static edu.cmu.dynet.internal.dynet_swig.lookup;
import static edu.cmu.dynet.internal.dynet_swig.parameter;

import java.util.Map;

import edu.cmu.dynet.internal.Dim;
import edu.cmu.dynet.internal.Expression;
import edu.cmu.dynet.internal.ExpressionVector;
import edu.cmu.dynet.internal.LongVector;
import edu.cmu.dynet.internal.LookupParameter;
import edu.cmu.dynet.internal.Parameter;
import edu.cmu.dynet.internal.ParameterCollection;
import edu.cmu.dynet.internal.SimpleRNNBuilder;

public class BidirectionalLSTM extends NeuralNetworkCore {
	
	private static final long serialVersionUID = 4592893499307238510L;

	private int inputDim;
	private int hiddenDim;
	private LookupParameter ltp;
	private Parameter r;
	private Parameter bias;
	private SimpleRNNBuilder fw;
	private SimpleRNNBuilder bw;
	int vocabSize;
	Map<String, Integer> word2int;
	public static String UNK = "<unk>";
	public static String PAD = "<pad>";
	
	private int maxLen;
	
	static Dim makeDim(int[] dims) {
	    LongVector dimInts = new LongVector();
	    for (int i = 0; i < dims.length; i++) {
	      dimInts.add(dims[i]);
	    }
	    return new Dim(dimInts);
	}
	
	public BidirectionalLSTM(int hiddenSize, int numLabels, Map<String, Integer> word2int) {
		super(numLabels);
		this.inputDim = hiddenSize;
		this.word2int = word2int;
		System.out.println(this.word2int);
		this.vocabSize = this.word2int.size();
	}

	public void setMaxLen(int maxLen) {
		System.err.println("[info] setting max len: " + maxLen);
		this.maxLen = maxLen;
	}
	
	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		return edgeInput;
	}
	
	@Override
	public int hyperEdgeInput2OutputRowIndex (Object edgeInput, NNDataHelper helper) {
		return helper.getNNInputId(edgeInput);
	}

	@Override
	public ParameterCollection initSpecificModelParam() {
		ParameterCollection model = new ParameterCollection();
		ltp =  model.add_lookup_parameters(vocabSize, makeDim(new int[]{inputDim}));
		r = model.add_parameters(makeDim(new int[]{this.numOutputs, inputDim}));
		bias = model.add_parameters(makeDim(new int[]{this.numOutputs}));
		return model;
	}

	@Override
	public Expression buildForwardGraph(Object[] inputs) {
//		System.out.println("size of inputs : " + inputs.size());
		Expression re = parameter(cg, r);
		Expression be = parameter(cg, bias);
		
		ExpressionVector f = new ExpressionVector();
		for(Object input : inputs) {
			String word = (String) input;
			int id = this.word2int.get(word);
			Expression cur = lookup(cg, ltp, id);
			Expression res = exprPlus(exprTimes(re, cur), be);
			f.add(res);
		}
		
		return concatenate(f);
	}



}
