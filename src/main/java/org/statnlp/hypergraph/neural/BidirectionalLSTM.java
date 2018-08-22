package org.statnlp.hypergraph.neural;

import static edu.cmu.dynet.internal.dynet_swig.*;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.List;
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
		this.hiddenDim = hiddenSize;
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
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		return sentAndPos.getKey();
	}
	
	@Override
	public int hyperEdgeInput2OutputRowIndex (Object edgeInput) {
		@SuppressWarnings("unchecked")
		SimpleImmutableEntry<String, Integer> sentAndPos = (SimpleImmutableEntry<String, Integer>) edgeInput;
		int sentID = this.getNNInputID(sentAndPos.getKey()); 
		int row = sentAndPos.getValue() * this.getNNInputSize() + sentID;
		return row;
	}

	@Override
	public ParameterCollection initalizeModelParams() {
		ParameterCollection model = new ParameterCollection();
		ltp =  model.add_lookup_parameters(vocabSize, makeDim(new int[]{inputDim}));
		fw = new SimpleRNNBuilder(1, inputDim, hiddenDim, model);
		bw = new SimpleRNNBuilder(1, inputDim, hiddenDim, model);
		r = model.add_parameters(makeDim(new int[]{this.numOutputs, 2 * hiddenDim}));
		bias = model.add_parameters(makeDim(new int[]{this.numOutputs}));
		return model;
	}

	@Override
	public Expression buildForwardGraph(List<Object> inputs) {
//		System.out.println("size of inputs : " + inputs.size());
		Expression re = parameter(cg, r);
		Expression be = parameter(cg, bias);
		String[][] sents = new String[inputs.size()][];
		for (int i = 0; i < inputs.size(); i++) {
			sents[i] = ((String)inputs.get(i)).split(" ");
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
//			System.out.println(curr.dim().batch_size());
//			System.out.println(curr.dim().rows() + " " + curr.dim().cols());
//			System.out.println(curr.dim().size());
//			Expression curr = lookup(cg, ltp, id); //for element
			Expression curr_y = fw.add_input(curr);
			finalForwardErr.add(curr_y);
		}
		
		bw.new_graph(cg);
		bw.start_new_sequence();
		ExpressionVector finalBackwardErr = new ExpressionVector();
		for (int i = this.maxLen-1 ; i >=0; i--) {
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
			Expression curr_y = bw.add_input(curr);
			finalBackwardErr.add(curr_y);
		}
//		for (int i = sent.length - 1 ; i >=0; i--) {
//			String word = sent[i];
//			long id = -1;
//			if (this.word2int.containsKey(word)) {
//				id = this.word2int.get(word);
//			} else {
//				id = this.word2int.get(UNK);
//			}
//			Expression curr = lookup(cg, ltp, id);
//			Expression curr_y = bw.add_input(curr);
//			finalBackwardErr.add(curr_y);
//		}
		ExpressionVector finalErr = new ExpressionVector();
		for (int i = 0; i < this.maxLen; i++) {
			Expression f = finalForwardErr.get(i);
			Expression b = finalForwardErr.get(this.maxLen - i -1);
			ExpressionVector ev = new ExpressionVector();
			ev.add(f);
			ev.add(b);
			Expression res = concatenate(ev);
			Expression curr_r = exprPlus(exprTimes(re, res), be);
			finalErr.add(curr_r);
		}
		
		Expression result = concatenate_cols(finalErr);
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
