package org.statnlp.example.semi_crf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Sentence;
import org.statnlp.commons.types.WordToken;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.GenerativeNetworkModel;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkConfig.ModelType;
import org.statnlp.hypergraph.NetworkModel;


public class SemiCRFMain {
	
	
	public static int trainNum = 1000;
	public static int testNumber = -1;
	public static int numThread = 8;
	public static int numIterations = 5000;
	public static double l2 = 0.01;
	public static String modelFile = null;
	public static boolean isTrain = true;
	public static String train_filename = "data/alldata/nbc/ecrf.train.MISC.txt";
	public static String test_filename = "data/alldata/nbc/ecrf.test.MISC.txt";
	/** true means using the predicted dependency features.. if not used dep features, this option does not matter**/
	
	
	private static void processArgs(String[] args) throws FileNotFoundException{
		for(int i=0;i<args.length;i=i+2){
			switch(args[i]){
				case "-trainNum": trainNum = Integer.valueOf(args[i+1]); break;   //default: all 
				case "-testNum": testNumber = Integer.valueOf(args[i+1]); break;    //default:all
				case "-iter": numIterations = Integer.valueOf(args[i+1]); break;   //default:100;
				case "-thread": numThread = Integer.valueOf(args[i+1]); break;   //default:5
				case "-windows": SemiEval.windows = true; break;            //default: false (is using windows system to run the evaluation script)
				case "-batch": NetworkConfig.USE_BATCH_TRAINING = true;
								NetworkConfig.BATCH_SIZE = Integer.valueOf(args[i+1]); break;
				case "-model": NetworkConfig.MODEL_TYPE = args[i+1].equals("crf")? ModelType.CRF:ModelType.SSVM;   break;
				case "-neural": if(args[i+1].equals("true")){ 
										NetworkConfig.USE_NEURAL_FEATURES = true; 
										NetworkConfig.REGULARIZE_NEURAL_FEATURES = false;
										NetworkConfig.OPTIMIZE_NEURAL = false;  //not optimize in CRF..
										NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
									}
								break;
				case "-reg": l2 = Double.valueOf(args[i+1]);  break;
				case "-modelPath": modelFile = args[i+1]; break;
				case "-mode": isTrain = args[i+1].equals("train")?true:false; break;
				case "-trainFile": train_filename = args[i+1]; break;
				case "-testFile": test_filename = args[i+1]; break;
				default: System.err.println("Invalid arguments, please check usage."); System.exit(0);
			}
		}
	}
	
	
	public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
		
		
		//always use conll data
//		train_filename = "data/semi/semi.train.txt";
//		test_filename = "data/semi/semi.test.txt";
		processArgs(args);
		/**data is 0-indexed, network compiler is 1-indexed since we have leaf nodes.**/
		/**Read the all data**/
		train_filename = "data/semi_example/debug.txt";
		test_filename = "data/semi_example/debug.txt";
		String resEval = "data/semi_example/resEval.txt";
		System.out.println("[Info] Reading data:"+train_filename);
		System.out.println("[Info] Reading data:"+test_filename);
		SemiCRFInstance[] trainInstances = readCoNLLData(train_filename, true,	trainNum);
		SemiCRFInstance[] testInstances	 = readCoNLLData(test_filename, false,	testNumber);
		
		/****Printing the total Number of entities***/
		int totalNumber = 0;
		int tokenNum = 0;
		for(SemiCRFInstance inst: trainInstances){
			totalNumber+=totalEntities(inst);
			tokenNum += inst.size();
		}
		System.out.println("[Info] Total number of entities in training:"+totalNumber+" token:"+tokenNum);
		totalNumber = 0;
		tokenNum = 0;
		for(SemiCRFInstance inst: testInstances){
			totalNumber+=totalEntities(inst);
			tokenNum += inst.size();
		}
		System.out.println("[Info] Total number of entities in testing:"+totalNumber+" token:"+tokenNum);
		/****(END) Printing the total Number of entities***/
		
		int maxSize = 0;
		int maxSpan = 0;
		for(SemiCRFInstance instance: trainInstances){
			maxSize = Math.max(maxSize, instance.size());
			for(Span span: instance.output){
				maxSpan = Math.max(maxSpan, span.end-span.start+1);
			}
		}
		for(SemiCRFInstance instance: testInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThread;

		//modify this. and read neural config
		
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		GlobalNetworkParam gnp = null;
		if(isTrain || modelFile==null || modelFile.equals("") || !new File(modelFile).exists()){
			gnp = new GlobalNetworkParam();
		}else{
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(modelFile));
			gnp=(GlobalNetworkParam)in.readObject();
			in.close();
		}
		
		SemiCRFNetworkCompiler compiler = new SemiCRFNetworkCompiler(maxSize, maxSpan);
		SemiCRFFeatureManager fm = new SemiCRFFeatureManager(gnp);
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler) : DiscriminativeNetworkModel.create(fm, compiler);
		
		
		if(isTrain){
			model.train(trainInstances, numIterations);
			if(modelFile!=null && !modelFile.equals("")){
				ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(modelFile));
				out.writeObject(fm.getParam_G());
				out.close();
			}
		}
		Instance[] predictions = model.decode(testInstances);
		SemiEval.evalNER(predictions, resEval);
		//SemiEval.writeNERResult(predictions, resRes);
		
		
	}
	
	/**
	 * Read data from file in a CoNLL format 0-index.
	 * @param fileName
	 * @param isLabeled
	 * @param isPipe: true means read the predicted features. always set to false for reading training instances.
	 * @return
	 * @throws IOException
	 */
	@SuppressWarnings("resource")
	private static SemiCRFInstance[] readCoNLLData(String fileName, boolean isLabeled, int number) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<SemiCRFInstance> result = new ArrayList<SemiCRFInstance>();
		ArrayList<WordToken> wts = new ArrayList<WordToken>();
		List<Span> output = new ArrayList<Span>();
		int instanceId = 1;
		int start = -1;
		int end = 0;
		Label prevLabel = null;
		int sentIndex = 0;
		int index = 0;
		while(br.ready()){
			String line = br.readLine().trim();
			if(line.length() == 0){
				sentIndex++;
				end = wts.size()-1;
				if(start != -1){
					createSpan(output, start, end, prevLabel);
				}
				SemiCRFInstance instance = new SemiCRFInstance(instanceId, 1.0);
				WordToken[] wtArr = new WordToken[wts.size()];
				instance.input = new Sentence(wts.toArray(wtArr));
				instance.output = output;
				/** debug information **/
				int realE = 0;
				for(int i=0;i<instance.input.length(); i++){
					if(instance.input.get(i).getEntity().startsWith("B-")) realE++;
				}
				int outputNum = 0;
				for(int i=0; i<instance.output.size();i++){
					if(!instance.output.get(i).label.form.equals("O")) outputNum++;
				}
				if(realE!=outputNum) {
					throw new RuntimeException("real number of entities:"+realE+" "+"span num:"+outputNum+" \n sent:"+sentIndex);
				}
				/***/
				//instance.leftDepRel = sent2LeftDepRel(instance.input);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				wts = new ArrayList<WordToken>();
				output = new ArrayList<Span>();
				prevLabel = null;
				start = -1;
				end = 0;
				index = 0;
				if(result.size()==number)
					break;
			} else {
				String[] values = line.split("[\t ]");
				//int index = Integer.valueOf(values[0]) - 1; //because it is starting from 1
				String word = values[0];
				String form = values[1];
				WordToken wt = new WordToken(word);
				wt.setEntity(form);
				wts.add(wt);
				Label label = null;
				if(form.startsWith("B")){
					if(start != -1){
						end = index - 1;
						createSpan(output, start, end, prevLabel);
					}
					start = index;
					label = Label.get(form.substring(2));
					
				} else if(form.startsWith("I")){
					label = Label.get(form.substring(2));
				} else if(form.startsWith("O")){
					if(start != -1){
						end = index - 1;
						createSpan(output, start, end, prevLabel);
					}
					start = -1;
					createSpan(output, index, index, Label.get("O"));
					label = Label.get("O");
				}
				prevLabel = label;
				index++;
			}
		}
		br.close();
		String type = isLabeled? "train":"test";
		System.out.println("[Info] number of "+type+" instances:"+result.size());
		return result.toArray(new SemiCRFInstance[result.size()]);
	}
	
 	private static void createSpan(List<Span> output, int start, int end, Label label){
		if(label==null){
			throw new RuntimeException("The label is null");
		}
		if(start>end){
			throw new RuntimeException("start cannot be larger than end");
		}
		if(label.form.equals("O")){
			for(int i=start; i<=end; i++){
				output.add(new Span(i, i, label));
			}
		} else {
			output.add(new Span(start, end, label));
		}
	}
	
	private static int totalEntities(SemiCRFInstance inst){
		int total = 0;
		List<Span> output = inst.getOutput();
//		Sentence sent = inst.getInput();
//		for(int i=0;i<sent.length();i++){
//			if(sent.get(i).getEntity().startsWith("B")) total++;
//		}
		for(Span span: output){
			Label label = span.label;
			if(label.equals(Label.get("O"))) continue;
			total++;
		}
		return total;
	}

}
