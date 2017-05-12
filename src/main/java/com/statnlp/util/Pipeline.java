/**
 *
 */
package com.statnlp.util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.Logger;

import com.statnlp.commons.ml.opt.GradientDescentOptimizer;
import com.statnlp.commons.ml.opt.OptimizerFactory;
import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.GenerativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkConfig.ModelType;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.util.instance_parser.InstanceParser;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.Argument;
import net.sourceforge.argparse4j.inf.ArgumentAction;
import net.sourceforge.argparse4j.inf.ArgumentChoice;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

/**
 * The generic main class to all StatNLP models.
 */
public abstract class Pipeline {
	
	public static final Logger LOGGER = GeneralUtils.createLogger(Pipeline.class);
	
	public InstanceParser instanceParser;
	public NetworkCompiler compiler;
	public FeatureManager fm;
	public NetworkModel networkModel;
	public GlobalNetworkParam param;
	
	public long timer = 0;
	
	private String currentTaskName;

	// Argument Parser-related class members
	public Namespace parameters;
	public String[] unprocessedArgs;
	public ArgumentParser argParser;
	protected HashMap<String, Argument> argParserObjects = new HashMap<String, Argument>();
	
	// Main tasks
	public final String TASK_TRAIN = "train";
	public final String TASK_TUNE = "tune";
	public final String TASK_TEST = "test";
	public final String TASK_EVALUATE = "evaluate";
	public final String TASK_VISUALIZE = "visualize";
	
	// Auxiliary Tasks
	public final String TASK_LOAD_MODEL = "loadModel";
	public final String TASK_SAVE_MODEL = "saveModel";
	public final String TASK_READ_INSTANCES = "readInstances";
	public final String TASK_SAVE_PREDICTIONS = "savePredictions";
	public final String TASK_EXTRACT_FEATURES = "extractFeatures";
	public final String TASK_WRITE_FEATURES = "writeFeatures";
	
	/** The list of tasks which have been registered and can be run. */
	public final LinkedHashMap<String, Runnable> registeredTasks = new LinkedHashMap<String, Runnable>();
	
	/** The list storing the list of tasks to be executed in the current pipeline. */
	public List<String> taskList;
	
	public Pipeline(){
		taskList = new ArrayList<String>();
		initRegisteredTasks();
		initArgumentParser();
	}
	
	private void initRegisteredTasks(){
		// Main tasks
		registeredTasks.put(TASK_TRAIN, this::Train);
		registeredTasks.put(TASK_TUNE, this::Tune);
		registeredTasks.put(TASK_TEST, this::Test);
		registeredTasks.put(TASK_EVALUATE, this::Evaluate);
		registeredTasks.put(TASK_VISUALIZE, this::Visualize);
		
		// Auxiliary tasks
		registeredTasks.put(TASK_LOAD_MODEL, this::LoadModel);
		registeredTasks.put(TASK_SAVE_MODEL, this::SaveModel);
		registeredTasks.put(TASK_READ_INSTANCES, this::ReadInstances);
		registeredTasks.put(TASK_SAVE_PREDICTIONS, this::SavePredictions);
		registeredTasks.put(TASK_EXTRACT_FEATURES, this::ExtractFeatures);
		registeredTasks.put(TASK_WRITE_FEATURES, this::WriteFeatures);
	}
	
	/**
	 * Registers a task.
	 * @param taskName
	 * @param action
	 */
	protected void registerTask(String taskName, Runnable action){
		registeredTasks.put(taskName, action);
	}
	
	/**
	 * Whether a number is a double value.
	 * @param val
	 * @return
	 */
	private static boolean isDouble(String val){
		try{
			Double.valueOf(val);
			return true;
		} catch (NumberFormatException e){
			return false;
		}
	}
	
	@SuppressWarnings("unchecked")
	private static String[] argsAsArray(Object value){
		String[] args;
		if(value instanceof List){
			args = ((List<String>)value).toArray(new String[0]);
		} else {
			args = (String[])value;
		}
		return args;
	}
	
	private void initArgumentParser(){
		argParser = ArgumentParsers.newArgumentParser("StatNLP Framework: "+this.getClass().getSimpleName())
                .defaultHelp(true)
                .description("Execute the model.");
		
		// The tasks to be executed
		argParserObjects.put("tasks", argParser.addArgument("tasks")
				.type(String.class)
				.metavar("tasks")
				.choices(registeredTasks.keySet())
				.nargs("+")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						List<String> unknownTasks = new ArrayList<String>();
						for(String task: args){
							if(!registeredTasks.containsKey(task)){
								unknownTasks.add(task);
							}
							taskList.add(task);
						}
						if(!unknownTasks.isEmpty()){
							throw new ArgumentParserException("Unrecognized tasks: "+unknownTasks, argParser);
						}
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The list of tasks to be executed. The registered tasks are:\n"
						+ registeredTasks.keySet()));
		
		// Training Settings
		argParserObjects.put("--maxIter", argParser.addArgument("--maxIter")
				.type(Integer.class)
				.setDefault(1000)
				.help("The maximum number of training iterations."));
		argParserObjects.put("--modelType", argParser.addArgument("--modelType")
				.type(ModelType.class)
				.choices(ModelType.values())
				.setDefault(NetworkConfig.MODEL_TYPE)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.MODEL_TYPE = ModelType.valueOf(args[0].toUpperCase());
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The type of the model during training. "
						+ "Model types SSVM and SOFTMAX_MARGIN require cost function to be defined"));
		argParserObjects.put("--objtol", argParser.addArgument("--objtol")
				.type(Double.class)
				.setDefault(NetworkConfig.OBJTOL)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.OBJTOL = Double.parseDouble(args[0]);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The minimum change in objective function to be considered not converged yet."));
		argParserObjects.put("--margin", argParser.addArgument("--margin")
				.type(Double.class)
				.setDefault(NetworkConfig.MARGIN)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.MARGIN = Double.parseDouble(args[0]);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The margin for margin-based methods (SSVM and SOFTMAX_MARGIN)."));
		argParserObjects.put("--weightInit", argParser.addArgument("--weightInit")
				.type(String.class)
				.setDefault(new String[]{"0.0"})
				.choices(new ArgumentChoice(){
					private final List<String> allowedArgs = Arrays.asList(new String[]{"random"});

					@Override
					public boolean contains(Object val) {
						try{
							if(isDouble((String)val)){
								return true;
							}
							if(allowedArgs.contains((String)val)){
								return true;
							}
						} catch (Exception e){
							return false;
						}
						return false;
					}

					@Override
					public String textualFormat() {
						return "?";
					}
					
				})
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						try{
							double initialWeight = Double.parseDouble((String)args[0]);
							NetworkConfig.FEATURE_INIT_WEIGHT = initialWeight;
						} catch (NumberFormatException e){
							if(args[0].equals("random")){
								NetworkConfig.RANDOM_INIT_WEIGHT = true;
								if(args.length > 1){
									NetworkConfig.RANDOM_INIT_FEATURE_SEED = Integer.parseInt(args[1]);
								}
							}
						}
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.nargs("+")
				.help("The margin for margin-based methods (SSVM and SOFTMAX_MARGIN).\n"
						+ "Use --weightInit <numeric> to specify an initial value to all weights,\n"
						+ "Use --weightInit random [optional_seed] to randomly assign values to all"
						+ "weights using the given seed."));
		argParserObjects.put("--useGenerativeModel", argParser.addArgument("--useGenerativeModel")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.TRAIN_MODE_IS_GENERATIVE = true;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to use generative model (like HMM) or discriminative model (like CRF)."));
		argParserObjects.put("--useGD", argParser.addArgument("--useGD")
				.action(Arguments.storeTrue())
				.help("Whether to use gradient descent (which will uses Adam with default parameters). "
						+ "Override the --useGD argument object programmatically to modify this default behavior."));
		argParserObjects.put("--useBatchTraining", argParser.addArgument("--useBatchTraining")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.USE_BATCH_TRAINING = true;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to use mini-batches during training."));
		argParserObjects.put("--batchSize", argParser.addArgument("--batchSize")
				.type(Integer.class)
				.setDefault(NetworkConfig.BATCH_SIZE)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.BATCH_SIZE = Integer.parseInt(args[0]);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The size of batch to be used."));
		argParserObjects.put("--l2", argParser.addArgument("--l2")
				.type(Double.class)
				.setDefault(NetworkConfig.L2_REGULARIZATION_CONSTANT)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.L2_REGULARIZATION_CONSTANT = Double.parseDouble(args[0]);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The L2 regularization value."));

		// Threading Settings
		argParserObjects.put("--nThreads", argParser.addArgument("--nThreads")
				.type(Integer.class)
				.setDefault(NetworkConfig.NUM_THREADS)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String[] args = argsAsArray(value);
						NetworkConfig.NUM_THREADS = Integer.parseInt(args[0]);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}
					
				})
				.help("The number of threads to be used."));
		
		// Feature Extraction
		argParserObjects.put("--serialTouch", argParser.addArgument("--serialTouch")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.PARALLEL_FEATURE_EXTRACTION = false;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to serialize the feature extraction process. "
						+ "By default the feature extraction is parallelized, which is faster."));
		argParserObjects.put("--touchLabeledOnly", argParser.addArgument("--touchLabeledOnly")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY = true;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to define the feature set based on the labeled data only. "
						+ "By default the feature set is created based on all possibilities "
						+ "(e.g., all possible transitions in linear-chain CRF vs only seen transitions)"));
		argParserObjects.put("--attemptMemorySaving", argParser.addArgument("--attemptMemorySaving")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.AVOID_DUPLICATE_FEATURES = true;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to attempt to reduce memory usage. "
						+ "The actual saving depends on the feature extractor implementation."));
		
		// Other Settings
		argParserObjects.put("--debugMode", argParser.addArgument("--debugMode")
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						NetworkConfig.DEBUG_MODE = true;
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return false;
					}
					
				})
				.help("Whether to enable debug mode."));
	}

	/**
	 * Initialize InstanceParser in the context of current pipeline
	 */
	protected abstract InstanceParser initInstanceParser();
	
	protected void initAndSetInstanceParser(){
		setInstanceParser(initInstanceParser());
	}
	
	public void setInstanceParser(InstanceParser instanceParser){
		this.instanceParser = instanceParser;
	}
	
	/**
	 * Initialize NetworkCompiler in the context of current pipeline
	 * It will call getNetworkCompilerParameters() from Parser to get necessary parameters
	 */
	protected abstract NetworkCompiler initNetworkCompiler();
	
	protected void initAndSetNetworkCompiler(){
		setNetworkCompiler(initNetworkCompiler());
	}
	
	public void setNetworkCompiler(NetworkCompiler compiler){
		this.compiler = compiler;
	}
	
	/**
	 * Initialize FeatureManager in the context of current pipeline
	 * It will call getFeatureMgrParameters() from Parser to get necessary parameters
	 */
	protected abstract FeatureManager initFeatureManager();
	
	protected void initAndSetFeatureManager(){
		setFeatureManager(initFeatureManager());
	}
	
	public void setFeatureManager(FeatureManager fm){
		this.fm = fm;
	}
	
	/**
	 * Initialize training stuff, including argument parsing, variable initialization
	 */
	protected abstract void initTraining();
	
	/**
	 * Initialize tuning stuff, including argument parsing, variable initialization
	 */
	protected abstract void initTuning();
	
	/**
	 * Initialize testing stuff, including argument parsing, variable initialization
	 */
	protected abstract void initTesting();
	
	/**
	 *  Initialize evaluation stuff, including argument parsing, variable initialization
	 */
	protected abstract void initEvaluation();
	
	/**
	 *  Initialize visualization stuff, including argument parsing, variable initialization
	 */
	protected abstract void initVisualization();
	
	protected abstract Instance[] getInstancesForTraining();
	protected abstract Instance[] getInstancesForTuning();
	protected abstract Instance[] getInstancesForTesting();
	protected abstract Instance[] getInstancesForEvaluation();
	protected abstract Instance[] getInstancesForVisualization();
	
	/**
	 * Train the model on the given training instances
	 * @param trainInstances
	 */
	protected abstract void train(Instance[] trainInstances);

	/**
	 * Tune the model on the given development instances.
	 * @param devInstances
	 */
	protected abstract void tune(Instance[] devInstances);

	/**
	 * Test the model on the test instances
	 * @param testInstances
	 */
	protected abstract void test(Instance[] testInstances);

	/**
	 * Evaluate the prediction performance
	 * @param output
	 */
	protected abstract void evaluate(Instance[] output);
	
	/**
	 * Visualize the given instances
	 * @param instances Instances to be visualized
	 */
	protected abstract void visualize(Instance[] instances);
	
	/**
	 * Save the trained model into disk
	 * @throws IOException
	 */
	protected abstract void saveModel() throws IOException;

	/**
	 * Load the trained model into memory
	 * @throws IOException
	 */
	protected abstract void loadModel() throws IOException;

	/**
	 * The task of saving predictions
	 */
	protected abstract void savePredictions();
	
	/**
	 * The task of just extracting features
	 */
	protected abstract void extractFeatures(Instance[] instances);
	
	/**
	 * The task of just writing features to file
	 */
	protected abstract void writeFeatures(Instance[] instances);

	/**
	 * Returns the optimizer factory to be used during training<br>
	 * By default uses L-BFGS for those learning algorithm using softmax and not using batch, and
	 * in other cases uses gradient descent with AdaM with default hyperparameters, which stops
	 * after seeing no progress after {@link GradientDescentOptimizer#DEFAULT_MAX_STAGNANT_ITER_COUNT}
	 * iterations.
	 * @return
	 */
	protected OptimizerFactory getOptimizerFactory(){
		if(NetworkConfig.MODEL_TYPE.USE_SOFTMAX && NetworkConfig.USE_BATCH_TRAINING == false){
			return OptimizerFactory.getLBFGSFactory();
		} else {
			return OptimizerFactory.getGradientDescentFactoryUsingAdaMThenStop();
		}
	}
	
	/**
	 * Initialize the {@link GlobalNetworkParam} object
	 */
	protected void initGlobalNetworkParam(){
		this.param = new GlobalNetworkParam(getOptimizerFactory());
	}
	
	protected abstract void handleSaveModelError(Exception e);
	
	protected abstract void handleLoadModelError(Exception e);

	protected void initNetworkModel() {
		if(param == null){
			initGlobalNetworkParam();
		}
		if(fm == null){
			initAndSetFeatureManager();
		}
		if(compiler == null){
			initAndSetNetworkCompiler();
		}
		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
			networkModel = GenerativeNetworkModel.create(fm, compiler);
		} else {
			networkModel = DiscriminativeNetworkModel.create(fm, compiler);
		}
	}
	
	public void Train(){
		//defined by user
		initTraining();
		
		//defined by user
		initAndSetInstanceParser();
		
		Instance[] trainInstances = getInstancesForTraining();
		
		initGlobalNetworkParam();
		initAndSetNetworkCompiler();
		initAndSetFeatureManager();
		
		initNetworkModel();
		
		train(trainInstances);
		
		SaveModel();
	}

	public void Tune(){
	}
	
	public void Test(){
		//defined by user
		initTesting();
		
		LoadModel();

		initAndSetInstanceParser();
		initAndSetNetworkCompiler();
		initAndSetFeatureManager();

		Instance[] testInstances = getInstancesForTesting();
		for(int k = 0; k < testInstances.length; k++){
			testInstances[k].setUnlabeled();
		}
		
		test(testInstances);
	}
	
	public void Evaluate(){
		initEvaluation();
		Instance[] instanceForEvaluation = getInstancesForEvaluation();
		evaluate(instanceForEvaluation);
	}
	
	public void Visualize(){
		initVisualization();
		Instance[] instances = getInstancesForVisualization();
		visualize(instances);
	}
	
	public void LoadModel() {
		try {
			//defined by user
			loadModel();
		} catch (IOException e) {
			handleLoadModelError(e);
		}
	}

	public void SaveModel() {
		try {
			saveModel();
		} catch (IOException e) {
			handleSaveModelError(e);
		}
	}
	
	public void ReadInstances(){
		getInstancesForTraining();
		getInstancesForTuning();
		getInstancesForTesting();
	}
	
	public void SavePredictions(){
		savePredictions();
	}
	
	public void ExtractFeatures(){
		Instance[] instances;
		instances = getInstancesForTraining();
		extractFeatures(instances);
		instances = getInstancesForTuning();
		extractFeatures(instances);
		instances = getInstancesForTesting();
		extractFeatures(instances);
	}
	
	public void WriteFeatures(){
		Instance[] instances;
		instances = getInstancesForTraining();
		writeFeatures(instances);
		instances = getInstancesForTuning();
		writeFeatures(instances);
		instances = getInstancesForTesting();
		writeFeatures(instances);
	}

	public Pipeline parseArgs(String[] args){
		return parseArgs(args, true);
	}
	
	public Pipeline parseArgs(String[] args, boolean retainExistingState){
//		// If we want to support typo in arguments
//		String[] mainArgs = null;
//		String[] restArgs = null;
//		for(int i=0; i<args.length; i++){
//			if(args[i].equals("--")){
//				mainArgs = new String[i];
//				restArgs = new String[args.length-i];
//				for(int j=0; j<i; j++){
//					mainArgs[j] = args[j];
//				}
//				for(int j=i+1; j<args.length; j++){
//					restArgs[j-i-1] = args[j];
//				}
//			}
//		}
//		if(mainArgs == null){
//			mainArgs = args;
//			restArgs = new String[0];
//		}
//    	parameters = argParser.parseArgsOrFail(mainArgs);
//    	unprocessedArgs = restArgs;
    	
		List<String> unknownArgs = new ArrayList<String>();
		try{
			if(retainExistingState && parameters != null){
				Namespace newParameters = argParser.parseKnownArgs(args, unknownArgs);
				for(String key: parameters.getAttrs().keySet()){
					parameters.getAttrs().put(key, newParameters.get(key));
				}
			} else {
				parameters = argParser.parseKnownArgs(args, unknownArgs);
			}
		} catch (ArgumentParserException e){
			LOGGER.error(argParser.formatHelp());
			LOGGER.error(e);
			System.exit(1);
		}
		unprocessedArgs = unknownArgs.toArray(new String[unknownArgs.size()]);
		return this;
	}
	
	protected void setCurrentTask(String task){
		currentTaskName = task;
	}
	
	protected String getCurrentTask(){
		return currentTaskName;
	}
	
	public void execute(){
		for(String task: taskList){
			Runnable action = registeredTasks.get(task);
			setCurrentTask(task);
			action.run();
		}
	}
	
	public void execute(String[] args){
		execute(args, true);
	}
	
	public void execute(String[] args, boolean retainExistingState) {
		parseArgs(args, retainExistingState);
		execute();
	}
	
	@SuppressWarnings("unchecked")
	public <T extends FeatureManager> T getFeatureManager(){
		return (T) this.fm;
	}
	
	@SuppressWarnings("unchecked")
	public <T extends NetworkCompiler> T getNetworkCompiler(){
		return (T) this.compiler;
	}
	
	public NetworkModel getNetworkModel(){
		return this.networkModel;
	}
	
	public void resetTimer(){
		this.timer = 0;
	}
	
	public long getTimer(){
		return this.timer;
	}
	
	public boolean hasParameter(String key){
		return this.parameters.getAttrs().containsKey(key);
	}
	
	public <T> T getParameter(String key){
		return this.parameters.get(key);
	}
	
	public void setParameter(String key, Object value){
		this.parameters.getAttrs().put(key, value);
	}
	
}
