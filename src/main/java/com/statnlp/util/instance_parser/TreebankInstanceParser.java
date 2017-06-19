/**
 * 
 */
package com.statnlp.util.instance_parser;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import com.statnlp.example.tree_crf.BinaryTree;
import com.statnlp.example.tree_crf.CNFRule;
import com.statnlp.example.tree_crf.Label;
import com.statnlp.example.tree_crf.TreeCRFInstance;
import com.statnlp.util.Pipeline;

public class TreebankInstanceParser extends InstanceParser {

	private static final long serialVersionUID = -4678360022097844920L;
	
	public Label rootLabel;
	public List<Label> labels;
	public Map<Label, Set<CNFRule>> rules;

	/**
	 * @param pipeline
	 */
	public TreebankInstanceParser(Pipeline<?> pipeline) {
		super(pipeline);
		labels = new ArrayList<Label>();
		rules = new HashMap<Label, Set<CNFRule>>();
		rootLabel = Label.get("ROOT");
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.instance_parser.InstanceParser#buildInstances(java.lang.String[])
	 */
	@Override
	public TreeCRFInstance[] buildInstances(String... sources) throws FileNotFoundException {
		Scanner br = new Scanner(new File(sources[0]), "UTF-8");
		ArrayList<TreeCRFInstance> result = new ArrayList<TreeCRFInstance>();
		int instanceId = 1;
		while(br.hasNextLine()){
			String line = br.nextLine();
			BinaryTree tree = BinaryTree.parse(line);
			TreeCRFInstance instance = new TreeCRFInstance(instanceId, 1);
			instanceId++;
			instance.input = tree.getWords();
			instance.output = tree;
			getRules(tree);
			result.add(instance);
		}
		br.close();
		labels.addAll(Label.LABELS.values());
		return result.toArray(new TreeCRFInstance[result.size()]);
	}
	
	private void getRules(BinaryTree tree){
		if(tree.left == null) return;
		Label leftSide = tree.value.label;
		Label firstRight = tree.left.value.label;
		Label secondRight = tree.right.value.label;
		CNFRule rule = new CNFRule(leftSide, firstRight, secondRight);
		if(!rules.containsKey(leftSide)){
			rules.put(leftSide, new HashSet<CNFRule>());
		}
		rules.get(leftSide).add(rule);
		getRules(tree.left);
		getRules(tree.right);
	}

}
