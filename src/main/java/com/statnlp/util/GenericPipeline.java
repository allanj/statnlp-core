package com.statnlp.util;

import static com.statnlp.util.GeneralUtils.sorted;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;

import com.statnlp.commons.types.Instance;
import com.statnlp.commons.types.LinearInstance;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.hybridnetworks.NetworkModel.TrainingIterationInformation;
import com.statnlp.hybridnetworks.TemplateBasedFeatureManager;
import com.statnlp.ui.visualize.type.VisualizationViewerEngine;
import com.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import com.statnlp.util.instance_parser.InstanceParser;

import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.Argument;
import net.sourceforge.argparse4j.inf.ArgumentAction;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;

public class GenericPipeline extends Pipeline<GenericPipeline> {

	public static final Logger LOGGER = GeneralUtils.createLogger(GenericPipeline.class);

	public GenericPipeline() {
		// Various Paths
		argParserObjects.put("--linearModelClass", argParser.addArgument("--linearModelClass")
				.type(String.class)
				.setDefault("com.statnlp.example.linear_crf.LinearCRF")
				.help("The class name of the model to be loaded (e.g., LinearCRF).\n"
						+ "Note that this generic pipeline assumes linear instances."));
		argParserObjects.put("--useFeatureTemplate", argParser.addArgument("--useFeatureTemplate")
				.type(Boolean.class)
				.action(Arguments.storeTrue())
				.help("Whether to use feature template when extracting features."));
		argParserObjects.put("--featureTemplatePath", argParser.addArgument("--featureTemplatePath")
				.type(String.class)
				.help("The path to feature template file."));
		argParserObjects.put("--trainPath", argParser.addArgument("--trainPath")
				.type(String.class)
				.help("The path to training data."));
		argParserObjects.put("--numTrain", argParser.addArgument("--numTrain")
				.type(Integer.class)
				.help("The number of training data to be taken from the training file."));
		argParserObjects.put("--devPath", argParser.addArgument("--devPath")
				.type(String.class)
				.help("The path to development data"));
		argParserObjects.put("--numDev", argParser.addArgument("--numDev")
				.type(Integer.class)
				.help("The number of development data to be taken from the development file."));
		argParserObjects.put("--testPath", argParser.addArgument("--testPath")
				.type(String.class)
				.help("The path to test data"));
		argParserObjects.put("--numTest", argParser.addArgument("--numTest")
				.type(Integer.class)
				.help("The number of test data to be taken from the test file."));
		argParserObjects.put("--modelPath", argParser.addArgument("--modelPath")
				.type(String.class)
				.help("The path to the model"));
		argParserObjects.put("--logPath", argParser.addArgument("--logPath")
				.type(String.class)
				.action(new ArgumentAction(){

					@Override
					public void run(ArgumentParser parser, Argument arg, Map<String, Object> attrs, String flag,
							Object value) throws ArgumentParserException {
						String logPath = (String)value;
						withLogPath(logPath);
					}

					@Override
					public void onAttach(Argument arg) {}

					@Override
					public boolean consumeArgument() {
						return true;
					}

				})
				.help("The path to log all information related to this pipeline execution."));
		argParserObjects.put("--writeModelAsText", argParser.addArgument("--writeModelAsText")
				.type(Boolean.class)
				.action(Arguments.storeTrue())
				.help("Whether to additionally write the model as text with .txt extension."));
		argParserObjects.put("--resultPath", argParser.addArgument("--resultPath")
				.type(String.class)
				.help("The path to where we should store prediction results."));
		argParserObjects.put("--evaluateEvery", argParser.addArgument("--evaluateEvery")
				.type(Integer.class)
				.setDefault(0)
				.metavar("n")
				.help("Evaluate on development set every n iterations."));
	}

	public GenericPipeline withFeatureTemplate(boolean useFeatureTemplate){
		setFeatureManager(new TemplateBasedFeatureManager(param));
		return getThis();
	}

	public GenericPipeline withFeatureTemplate(boolean useFeatureTemplate, String featureTemplatePath){
		setFeatureManager(new TemplateBasedFeatureManager(param, featureTemplatePath));
		return this;
	}

	/**
	 * With the specified path to training data.
	 * @param trainPath
	 * @return
	 */
	public GenericPipeline withTrainPath(String trainPath){
		setParameter("trainPath", trainPath);
		return getThis();
	}

	/**
	 * With the specified path to devlopment data.
	 * @param devPath
	 * @return
	 */
	public GenericPipeline withDevPath(String devPath){
		setParameter("devPath", devPath);
		return getThis();
	}

	/**
	 * With the specified path to test data.
	 * @param testPath
	 * @return
	 */
	public GenericPipeline withTestPath(String testPath){
		setParameter("testPath", testPath);
		return getThis();
	}

	/**
	 * With the specified number of training data used.<br>
	 * If the number is more than the number of actual training data read from trainPath,
	 * all training data is used. 
	 * @param numTrain
	 * @return
	 */
	public GenericPipeline withNumTrain(int numTrain){
		setParameter("numTrain", numTrain);
		return getThis();
	}

	/**
	 * With the specified number of development data used.<br>
	 * If the number is more than the number of actual development data read from devPath,
	 * all development data is used. 
	 * @param numDev
	 * @return
	 */
	public GenericPipeline withNumDev(int numDev){
		setParameter("numDev", numDev);
		return getThis();
	}

	/**
	 * With the specified number of test data used.<br>
	 * If the number is more than the number of actual test data read from testPath,
	 * all test data is used. 
	 * @param numTest
	 * @return
	 */
	public GenericPipeline withNumTest(int numTest){
		setParameter("numTest", numTest);
		return getThis();
	}

	/**
	 * With the specified path to model.
	 * @param modelPath
	 * @return
	 */
	public GenericPipeline withModelPath(String modelPath){
		setParameter("modelPath", modelPath);
		return getThis();
	}

	/**
	 * With the specified path to the log file.
	 * @param logPath
	 * @return
	 */
	public GenericPipeline withLogPath(String logPath){
		setParameter("logPath", logPath);
		GeneralUtils.updateLogger(logPath);
		return getThis();
	}

	/**
	 * With the specified path to the result file.
	 * @param resultPath
	 * @return
	 */
	public GenericPipeline withResultPath(String resultPath){
		setParameter("resultPath", resultPath);
		return getThis();
	}

	/**
	 * Whether to also write the learned model as text, showing the feature list and the corresponding weights.
	 * @param writeModelAsText
	 * @return
	 */
	public GenericPipeline withWriteModelAsText(boolean writeModelAsText){
		setParameter("writeModelAsText", writeModelAsText);
		return getThis();
	}

	/**
	 * With evaluation on development data every specified number of iterations.
	 * @param evaluateEvery
	 * @return
	 */
	public GenericPipeline withEvaluateEvery(int evaluateEvery){
		setParameter("evaluateEvery", evaluateEvery);
		return getThis();
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initInstanceParser()
	 */
	@Override
	protected InstanceParser initInstanceParser() {
		if(instanceParser == null){
			if(instanceParserClass != null){
				try {
					return instanceParserClass.getConstructor(Pipeline.class).newInstance(this);
				} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
						| InvocationTargetException | NoSuchMethodException | SecurityException e) {
					LOGGER.fatal("[%s]Instance parser class name %s cannot be instantiated with Pipeline object as the argument.", getCurrentTask(), instanceParserClass);
					throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
				}
			}
			return new DelimiterBasedInstanceParser(this);
		} else {
			return instanceParser;
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initNetworkCompiler()
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected NetworkCompiler initNetworkCompiler() {
		if(networkCompiler != null){
			return networkCompiler;
		}
		String currentTask = getCurrentTask();
		if(currentTask.equals(TASK_TRAIN) || currentTask.equals(TASK_VISUALIZE)){
			if(networkCompilerClass == null){
				String linearModelClassName = getParameter("linearModelClass");
				String networkCompilerClassName = linearModelClassName+"NetworkCompiler";
				try {
					networkCompilerClass = (Class<? extends NetworkCompiler>)Class.forName(networkCompilerClassName);
				} catch (ClassNotFoundException e) {
					LOGGER.fatal("[%s]Network compiler class name cannot be inferred from model class name %s", getCurrentTask(), linearModelClassName);
					throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
				}
			}
			try {
				return (NetworkCompiler)networkCompilerClass.getConstructor(Pipeline.class).newInstance(this);
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
					InvocationTargetException | NoSuchMethodException | SecurityException e) {
				LOGGER.fatal("[%s]Network compiler class %s cannot be instantiated with Pipeline object as the argument.", getCurrentTask(), networkCompilerClass.getName());
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			}
		} else {
			if(networkModel == null){
				LOGGER.warn("[%s]No model has been loaded, cannot load network compiler.", getCurrentTask());
				return null;
			}
			return networkModel.getNetworkCompiler();
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initFeatureManager()
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected FeatureManager initFeatureManager() {
		if(featureManager != null){
			return featureManager;
		}
		String currentTask = getCurrentTask();
		if(currentTask.equals(TASK_TRAIN) || currentTask.equals(TASK_VISUALIZE)){
			if(featureManagerClass == null){
				if(hasParameter("useFeatureTemplate") && (boolean)getParameter("useFeatureTemplate")){
					if(hasParameter("featureTemplatePath")){
						return new TemplateBasedFeatureManager(param, (String)getParameter("featureTemplatePath"));	
					} else {
						return new TemplateBasedFeatureManager(param);
					}
				}
				String linearModelClassName = getParameter("linearModelClass");
				String featureManagerClassName = linearModelClassName+"FeatureManager";
				try {
					featureManagerClass = (Class<? extends FeatureManager>)Class.forName(featureManagerClassName);
				} catch (ClassNotFoundException e) {
					LOGGER.fatal("[%s]Feature manager class name cannot be inferred from model class name %s", getCurrentTask(), linearModelClassName);
					throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
				} 
			} else {
				if(hasParameter("useFeatureTemplate") && (boolean)getParameter("useFeatureTemplate")){
					LOGGER.warn("[%s]Both useFeatureTemplate and featureManagerClass are specified. Using the specified featureManagerClass.", getCurrentTask());
				}
			}
			try {
				return (FeatureManager)featureManagerClass.getConstructor(Pipeline.class).newInstance(this);
			} catch (InstantiationException | IllegalAccessException | IllegalArgumentException |
					InvocationTargetException | NoSuchMethodException | SecurityException e) {
				LOGGER.fatal("[%s]Feature manager class name cannot be inferred from model class name %s", getCurrentTask(), featureManagerClass.getName());
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			}
		} else {
			if(networkModel == null){
				LOGGER.warn("[%s]No model has been loaded, cannot load feature manager.", getCurrentTask());
				return null;
			}
			return networkModel.getFeatureManager();
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#saveModel()
	 */
	@Override
	protected void saveModel() throws IOException {
		String modelPath = getParameter("modelPath");
		if(modelPath == null){
			if(getCurrentTask() == TASK_SAVE_MODEL){
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Saving model requires --modelPath to be set."));
			} else {
				LOGGER.warn("[%s]Not saving trained model, since --modelPath is not set.", getCurrentTask());
				return;
			}
		}
		LOGGER.info("[%s]Writing model into %s...", getCurrentTask(), modelPath);
		long startTime = System.nanoTime();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
		oos.writeObject(networkModel);
		oos.writeObject(instanceParser);
		oos.close();
		long endTime = System.nanoTime();
		LOGGER.info("[%s]Writing model...Done in %.3fs", getCurrentTask(), (endTime-startTime)/1.0e9);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleSaveModelError(java.lang.Exception)
	 */
	@Override
	protected void handleSaveModelError(Exception e) {
		LOGGER.warn("[%s]Cannot save model to %s", getCurrentTask(), (String)getParameter("modelPath"));
		LOGGER.throwing(Level.WARN, e);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#loadModel()
	 */
	@Override
	protected void loadModel() throws IOException {
		String currentTask = getCurrentTask();
		if(networkModel != null && (currentTask.equals(TASK_TEST) || currentTask.equals(TASK_TUNE))){
			LOGGER.info("[%s]Model already loaded, using loaded model.", getCurrentTask());
		} else {
			String modelPath = getParameter("modelPath");
			if(modelPath == null){
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Loading model requires --modelPath to be set."));
			}
			LOGGER.info("Reading model from %s...", modelPath);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
			long startTime = System.nanoTime();
			try {
				networkModel = (NetworkModel)ois.readObject();
				instanceParser = (InstanceParser)ois.readObject();
			} catch (ClassNotFoundException e) {
				LOGGER.warn("[%s]Cannot load the model from %s", getCurrentTask(), modelPath);
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			} finally {
				ois.close();
			}
			long endTime = System.nanoTime();
			LOGGER.info("[%s]Reading model...Done in %.3fs", getCurrentTask(), (endTime-startTime)/1.0e9);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleLoadModelError(java.lang.Exception)
	 */
	@Override
	protected void handleLoadModelError(Exception e) {
		LOGGER.error("[%s]Cannot load model from %s", getCurrentTask(), (String)getParameter("modelPath"));
		throw new RuntimeException(LOGGER.throwing(Level.ERROR, e));
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTraining(java.lang.String[])
	 */
	@Override
	protected void initTraining() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTuning()
	 */
	@Override
	protected void initTuning() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initTesting()
	 */
	@Override
	protected void initTesting() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initEvaluation(java.lang.String[])
	 */
	@Override
	protected void initEvaluation() {
		// TODO Auto-generated method stub

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initVisualization()
	 */
	@Override
	protected void initVisualization() {
		if(instanceParser == null){
			if(hasParameter("trainPath")){
				initTraining();
			} else if(hasParameter("devPath")){
				initTuning();
			} else if(hasParameter("testPath")){
				initTesting();
			} else {
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("["+getCurrentTask()+"]Visualization requires one of --trainPath, --devPath, or --testPath to be specified."));
			}
		}
		initGlobalNetworkParam();

		initAndSetInstanceParser();

		if(hasParameter("trainPath")){
			getInstancesForTraining();
		} else if(hasParameter("devPath")){
			getInstancesForTuning();
		} else if(hasParameter("testPath")){
			getInstancesForTesting();
		}

		initAndSetNetworkCompiler();
		initAndSetFeatureManager();

		initNetworkModel();

	}

	protected Instance[] getInstancesForTraining(){
		if(hasParameter("trainInstances")){
			return getParameter("trainInstances");
		}
		if(!hasParameter("trainPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --trainPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] trainInstances = instanceParser.buildInstances((String)getParameter("trainPath"));
			if(hasParameter("numTrain")){
				int numTrain = getParameter("numTrain");
				if(numTrain > 0){
					numTrain = Math.min(trainInstances.length, numTrain);
					trainInstances = Arrays.copyOfRange(trainInstances, 0, numTrain);
				}
			}
			setParameter("trainInstances", trainInstances);
			return trainInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	protected Instance[] getInstancesForTuning(){
		if(hasParameter("devInstances")){
			return getParameter("devInstances");
		}
		if(!hasParameter("devPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --devPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] devInstances = instanceParser.buildInstances((String)getParameter("devPath"));
			if(hasParameter("numDev")){
				int numDev = getParameter("numDev");
				if(numDev > 0){
					numDev = Math.min(devInstances.length, numDev);
					devInstances = Arrays.copyOfRange(devInstances, 0, numDev);
				}
			}
			setParameter("devInstances", devInstances);
			return devInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	protected Instance[] getInstancesForTesting(){
		if(hasParameter("testInstances")){
			return getParameter("testInstances");
		}
		if(!hasParameter("testPath")){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("["+getCurrentTask()+"]The task %s requires --testPath to be specified.", getCurrentTask())));
		}
		try {
			Instance[] testInstances = instanceParser.buildInstances((String)getParameter("testPath"));
			if(hasParameter("numTest")){
				int numTest = getParameter("numTest");
				if(numTest > 0){
					numTest = Math.min(testInstances.length, numTest);
					testInstances = Arrays.copyOfRange(testInstances, 0, numTest);
				}
			}
			setParameter("testInstances", testInstances);
			return testInstances;
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#getInstancesForEvaluation()
	 */
	@Override
	protected Instance[] getInstancesForEvaluation() {
		return getParameter("testInstances");
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#getInstancesForEvaluation()
	 */
	@Override
	protected Instance[] getInstancesForVisualization() {
		Instance[] result = getParameter("trainInstances");
		if(result == null){
			result = getParameter("devInstances");
		}
		if(result == null){
			result = getParameter("testInstances");
		}
		if(result == null){
			throw LOGGER.throwing(new RuntimeException("["+getCurrentTask()+"]Cannot find instances to be visualized. "
					+ "Please specify them through --trainPath, --devPath, or --testPath."));
		}
		return result;
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#tune()
	 */
	@Override
	protected void train(Instance[] trainInstances) {
		if(hasParameter("evaluateEvery")){
			int evaluateEvery = getParameter("evaluateEvery");
			if(evaluateEvery > 0){
				networkModel.setEndOfIterCallback(new Consumer<TrainingIterationInformation>(){

					@Override
					public void accept(TrainingIterationInformation t) {
						int iterNum = t.iterNum;
						if((iterNum+1) % evaluateEvery == 0){
							Instance[] instances = getInstancesForTuning();
							for(int k = 0; k < instances.length; k++){
								instances[k].setUnlabeled();
							}

							try {
								instances = networkModel.decode(instances, true);
							} catch (InterruptedException e) {}
							evaluate(instances);
						}
					}

				});
			}
		}
		long duration = System.nanoTime();
		try {
			networkModel.train(trainInstances, getParameter("maxIter"));
			duration = System.nanoTime() - duration;
		} catch (InterruptedException e) {
			throw LOGGER.throwing(new RuntimeException(e));
		}
		LOGGER.info("[%s]Total training time: %.3fs\n", getCurrentTask(), duration/1.0e9);
		if((boolean)getParameter("writeModelAsText")){
			String modelPath = getParameter("modelPath");
			String modelTextPath = modelPath+".txt";
			try{
				LOGGER.info("[%s]Writing model text into %s...", getCurrentTask(), modelTextPath);
				PrintStream modelTextWriter = new PrintStream(modelTextPath);
				modelTextWriter.println(NetworkConfig.getConfig());
//				modelTextWriter.println("Model path: "+modelPath);
//				modelTextWriter.println("Train path: "+trainPath);
//				modelTextWriter.println("Test path: "+testPath);
//				modelTextWriter.println("#Threads: "+NetworkConfig.NUM_THREADS);
//				modelTextWriter.println("L2 param: "+NetworkConfig.L2_REGULARIZATION_CONSTANT);
//				modelTextWriter.println("Weight init: "+0.0);
//				modelTextWriter.println("objtol: "+NetworkConfig.OBJTOL);
//				modelTextWriter.println("Max iter: "+numIterations);
//				modelTextWriter.println();
//				modelTextWriter.println("Labels:");
//				List<Label> labelsUsed = new ArrayList<Label>(param.LABELS.values());
//				Collections.sort(labelsUsed);
//				modelTextWriter.println(labelsUsed);
				modelTextWriter.println("Num features: "+param.countFeatures());
				modelTextWriter.println("Features:");
				HashMap<String, HashMap<String, HashMap<String, Integer>>> featureIntMap = param.getFeatureIntMap();
				for(String featureType: sorted(featureIntMap.keySet())){
					modelTextWriter.println(featureType);
					HashMap<String, HashMap<String, Integer>> outputInputMap = featureIntMap.get(featureType);
					for(String output: sorted(outputInputMap.keySet())){
						modelTextWriter.println("\t"+output);
						HashMap<String, Integer> inputMap = outputInputMap.get(output);
						for(String input: sorted(inputMap.keySet())){
							int featureId = inputMap.get(input);
							modelTextWriter.printf("\t\t%s %d %.17f\n", input, featureId, featureManager.getParam_G().getWeight(featureId));
						}
					}
				}
				modelTextWriter.close();
			} catch (IOException e){
				LOGGER.warn("[%s]Cannot write model text into %s.", getCurrentTask(), modelTextPath);
				LOGGER.throwing(Level.WARN, e);
			}
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#tune()
	 */
	@Override
	protected void tune(Instance[] devInstances) {

	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#test()
	 */
	@Override
	protected void test(Instance[] testInstances) {
		long duration = System.nanoTime();
		try {
			Instance[] instanceWithPredictions = networkModel.decode(testInstances);
			duration = System.nanoTime() - duration;
			setParameter("testInstances", instanceWithPredictions);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		LOGGER.info("[%s]Total testing time: %.3fs\n", getCurrentTask(), duration/1.0e9);		
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#evaluationResult(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void evaluate(Instance[] instancesWithPrediction) {
		int corr = 0;
		int total = 0;
		for(Instance instance: instancesWithPrediction){
			LinearInstance<?> linInstance = (LinearInstance<?>)instance;
			try{
				corr += linInstance.countNumCorrectlyPredicted();
			} catch (IndexOutOfBoundsException e){
				throw new RuntimeException("This is usually caused by different number of predictions "
						+ "compared to gold. The default evaluation procedure assumes tagging task, "
						+ "with the same number of predictions. You can create custom evaluation procedure "
						+ "by either overriding the evaluate(Instance[]) function in a subclass of GenericPipeline, "
						+ "or supplying the evaluateCallback function through setEvaluateCallback.", e);
			}
			total += linInstance.size();
		}
		LOGGER.info("[%s]Correct/Total: %d/%d", getCurrentTask(), corr, total);
		LOGGER.info("[%s]Accuracy: %.2f%%", getCurrentTask(), 100.0*corr/total);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#visualize(com.statnlp.commons.types.Instance[])
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected void visualize(Instance[] instances) {
		if(visualizerClass == null){
			String visualizerModelName = getParameter("linearModelClass")+"Viewer";
			try{
				visualizerClass = (Class<VisualizationViewerEngine>)Class.forName(visualizerModelName);
			} catch (ClassNotFoundException e) {
				LOGGER.warn("[%s]Cannot automatically find viewer class for model name %s", getCurrentTask(), (String)getParameter("linearModelClass"));
				LOGGER.throwing(Level.WARN, e);
				return;
			}
		}
		try {
			networkModel.visualize(visualizerClass, instances);
		} catch (InterruptedException e) {
			LOGGER.info("[%s]Visualizer was interrupted.", getCurrentTask());
		}     
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#savePredictions()
	 */
	@Override
	protected void savePredictions() {
		if(!hasParameter("resultPath")){
			LOGGER.warn("[%s]Task savePredictions requires --resultPath to be specified.", getCurrentTask());
			return;
		}
		String resultPath = getParameter("resultPath");
		try{
			PrintWriter printer = new PrintWriter(new File(resultPath));
			Instance[] instances = getParameter("testInstances");
			for(Instance instance: instances){
				printer.println(instance.toString());
			}
			printer.close();
		} catch (FileNotFoundException e){
			LOGGER.warn("[%s]Cannot find file %s for storing prediction results.", getCurrentTask(), resultPath);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#extractFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void extractFeatures(Instance[] instances) {
		// TODO Auto-generated method stub
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#writeFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void writeFeatures(Instance[] instances) {
		// TODO Auto-generated method stub

	}

	public void initExecute(){
		boolean hasWarning = false;
		if(taskList.contains(TASK_TRAIN)){
			if(!hasParameter("trainPath")){
				LOGGER.warn("train task is specified but --trainPath is missing.");
				hasWarning = true;
			}
			if(!hasParameter("modelPath")){
				LOGGER.warn("Trained model will not be saved since --modelPath is missing.");
				hasWarning = true;
			}
		}
		if(taskList.contains(TASK_TUNE)){
			if(!hasParameter("devPath")){
				LOGGER.warn("tune task is specified but --devPath is missing.");
				hasWarning = true;
			}
		}
		if(taskList.contains(TASK_TEST)){
			if(!hasParameter("testPath")){
				LOGGER.warn("test task is specified but --testPath is missing.");
				hasWarning = true;
			}
		}
		if(hasWarning){
			try{Thread.sleep(2000);}catch(InterruptedException e){}
		}
	}

}
