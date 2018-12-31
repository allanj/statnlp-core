package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.concatenate_cols;
import static edu.cmu.dynet.internal.dynet_swig.exprPlus;
import static edu.cmu.dynet.internal.dynet_swig.exprTimes;
import static edu.cmu.dynet.internal.dynet_swig.lookup;
import static edu.cmu.dynet.internal.dynet_swig.parameter;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Map;

import edu.cmu.dynet.internal.Dim;
import edu.cmu.dynet.internal.Expression;
import edu.cmu.dynet.internal.ExpressionVector;
import edu.cmu.dynet.internal.LongVector;
import edu.cmu.dynet.internal.LookupParameter;
import edu.cmu.dynet.internal.Parameter;
import edu.cmu.dynet.internal.ParameterCollection;
import edu.cmu.dynet.internal.SimpleRNNBuilder;
import edu.cmu.dynet.internal.UnsignedVector;

public class BidirectionalLSTM extends NeuralNetworkCore {
	
	private static final long serialVersionUID = 4592893499307238510L;

	private int inputDim;
	private int hiddenDim;
	private LookupParameter ltp;
	private Parameter r;
	private Parameter bias;
	private SimpleRNNBuilder fw;
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
		this.inputDim = 5;
		this.hiddenDim = 5 ;
		System.out.println(this.word2int);
		this.vocabSize = this.word2int.size();
	}

	public void setMaxLen(int maxLen) {
		System.err.println("[info] setting max len: " + maxLen);
		this.maxLen = maxLen;
	}
	
	@Override
	public Object hyperEdgeInput2NNInput(Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		return sentAndPos.getKey();
	}
	
	@Override
	public int hyperEdgeInput2OutputRowIndex (Object edgeInput, NNDataHelper helper) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		int sentID = helper.getNNInputId(sentAndPos.getKey()); 
		int row = sentAndPos.getValue() * helper.getNNInputSize() + sentID;
		return row;
	}

	@Override
	public ParameterCollection initSpecificModelParam() {
		ParameterCollection model = new ParameterCollection();
		ltp =  model.add_lookup_parameters(vocabSize, makeDim(new int[]{inputDim}));
		fw = new SimpleRNNBuilder(1, inputDim, hiddenDim, model);
		r = model.add_parameters(makeDim(new int[]{this.numOutputs, hiddenDim}));
		bias = model.add_parameters(makeDim(new int[]{this.numOutputs}));
		return model;
	}

	@Override
	public Expression buildForwardGraph(Object[] inputs) {
//		System.out.println("size of inputs : " + inputs.size());
		Expression re = parameter(cg, r);
		Expression be = parameter(cg, bias);
		String[][] sents = new String[inputs.length][];
		for (int i = 0; i < inputs.length; i++) {
			sents[i] = ((String)inputs[i]).split(" ");
		}
		fw.new_graph(cg);
		fw.start_new_sequence();
		ExpressionVector finalForwardErr = new ExpressionVector();
		for (int i = 0 ; i < this.maxLen; i++) {
			UnsignedVector uv = new UnsignedVector();
			for (int s = 0; s < sents.length; s++) {
				long id = -1;
				if (i >= sents[s].length) {
					id = 0;
					uv.add(id);
					break;
				}
				String word = sents[s][i];
				if (this.word2int.containsKey(word)) {
					id = this.word2int.get(word);
				} else {
					id = this.word2int.get(UNK);
				}
				uv.add(id);
			}
			Expression curr = lookup(cg, ltp, uv);
			Expression curr_y = fw.add_input(curr);
			Expression curr_r = exprPlus(exprTimes(re, curr_y), be);
			finalForwardErr.add(curr_r);
		}
		Expression result = concatenate_cols(finalForwardErr);
//		System.out.println("result dimension : " + result.dim().ndims());
//		System.out.println("result size : " + result.dim().size());
//		System.out.println("batch size: " +result.dim().batch_size());
//		System.out.println("batch size: " +result.dim().batch_elems());
//		System.out.println("result dimension : " + result.dim().rows() + " " + result.dim().cols());
		
//		for (int i = 0; i < result.dim().size(); i++) {
//			System.out.println(result.dim().get(i));
//		}
//		System.out.println("forward pass");
		return result;
	}



}
