/**
 * 
 */
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.FileAppender;
import org.apache.logging.log4j.core.config.AppenderRef;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.logging.log4j.core.layout.PatternLayout;

import com.statnlp.commons.types.Instance;
import com.statnlp.commons.types.LinearInstance;
import com.statnlp.hybridnetworks.FeatureManager;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;
import com.statnlp.ui.visualize.type.VisualizationViewerEngine;
import com.statnlp.util.instance_parser.DelimiterBasedInstanceParser;
import com.statnlp.util.instance_parser.InstanceParser;

import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.Argument;
import net.sourceforge.argparse4j.inf.ArgumentAction;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;

public class GenericPipeline extends Pipeline{
	
	public static final Logger LOGGER = GeneralUtils.createLogger(GenericPipeline.class);

	public GenericPipeline() {
		// Various Paths
		argParserObjects.put("--linearModelClass", argParser.addArgument("--linearModelClass")
				.type(String.class)
				.setDefault("com.statnlp.example.linear_crf.LinearCRF")
				.help("The class name of the model to be loaded (e.g., LinearCRF).\n"
						+ "Note that this generic pipeline assumes linear instances."));
		argParserObjects.put("--trainPath", argParser.addArgument("--trainPath")
				.type(String.class)
				.help("The path to training data."));
		argParserObjects.put("--devPath", argParser.addArgument("--devPath")
				.type(String.class)
				.help("The path to development data"));
		argParserObjects.put("--testPath", argParser.addArgument("--testPath")
				.type(String.class)
				.help("The path to test data"));
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
						attrs.put("logPath", logPath);
						final LoggerContext ctx = (LoggerContext) LogManager.getContext(true);
				        final Configuration config = ctx.getConfiguration();
				        PatternLayout layout = PatternLayout.newBuilder()
				        					.withPattern("%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n")
				        					.withConfiguration(config)
				        					.build();
						FileAppender appender = ((org.apache.logging.log4j.core.util.Builder<FileAppender>)FileAppender.newBuilder()
				        					.withFileName(logPath)
				        					.withAppend(false)
				        					.withLocking(false)
				        					.withName("File")
				        					.withImmediateFlush(true)
				        					.withIgnoreExceptions(false)
				        					.withBufferedIo(false)
				        					.withBufferSize(4000)
				        					.withLayout(layout)
				        					.withAdvertise(false)
				        					.setConfiguration(config))
				        					.build();
				        appender.start();
				        config.addAppender(appender);
				        AppenderRef ref = AppenderRef.createAppenderRef("File", null, null);
				        AppenderRef[] refs = new AppenderRef[] {ref};
				        LoggerConfig loggerConfig = LoggerConfig.createLogger(false, Level.INFO, "org.apache.logging.log4j",
				            "true", refs, null, config, null );
				        loggerConfig.addAppender(appender, null, null);
				        config.addLogger("org.apache.logging.log4j", loggerConfig);
				        ctx.updateLoggers();
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
	}
	
	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initInstanceParser()
	 */
	@Override
	protected InstanceParser initInstanceParser() {
		if(instanceParser == null){
			return new DelimiterBasedInstanceParser(this);
		} else {
			return instanceParser;
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initNetworkCompiler()
	 */
	@Override
	protected NetworkCompiler initNetworkCompiler() {
		String currentTask = getCurrentTask();
		if(currentTask.equals(TASK_TRAIN) || currentTask.equals(TASK_VISUALIZE)){
			String linearModelClassName = getParameter("linearModelClass");
			String networkCompilerClassName = linearModelClassName+"NetworkCompiler";
			try {
				return (NetworkCompiler)Class.forName(networkCompilerClassName).getConstructor(Pipeline.class).newInstance(this);
			} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | 
					IllegalArgumentException | InvocationTargetException | NoSuchMethodException | 
					SecurityException e) {
				LOGGER.fatal("Network compiler class name cannot be inferred from model class name %s", linearModelClassName);
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			}
		} else {
			if(networkModel == null){
				LOGGER.warn("No model has been loaded, cannot load network compiler.");
				return null;
			}
			return networkModel.getNetworkCompiler();
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#initFeatureManager()
	 */
	@Override
	protected FeatureManager initFeatureManager() {
		String currentTask = getCurrentTask();
		if(currentTask.equals(TASK_TRAIN) || currentTask.equals(TASK_VISUALIZE)){
			String linearModelClassName = getParameter("linearModelClass");
			String featureManagerClassName = linearModelClassName+"FeatureManager";
			try {
				return (FeatureManager)Class.forName(featureManagerClassName).getConstructor(Pipeline.class).newInstance(this);
			} catch (ClassNotFoundException | InstantiationException | IllegalAccessException | 
					IllegalArgumentException | InvocationTargetException | NoSuchMethodException | 
					SecurityException e) {
				LOGGER.fatal("Feature manager class name cannot be inferred from model class name %s", linearModelClassName);
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			}
		} else {
			if(networkModel == null){
				LOGGER.warn("No model has been loaded, cannot load feature manager.");
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
			throw LOGGER.throwing(Level.ERROR, new RuntimeException("Saving model requires --modelPath to be set."));
		}
		LOGGER.info("Writing model into %s...", modelPath);
		long startTime = System.currentTimeMillis();
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
		oos.writeObject(networkModel);
		oos.writeObject(instanceParser);
		oos.close();
		long endTime = System.currentTimeMillis();
		LOGGER.info("Writing model...Done in %.3fs", (endTime-startTime)/1000.0);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleSaveModelError(java.lang.Exception)
	 */
	@Override
	protected void handleSaveModelError(Exception e) {
		LOGGER.warn("Cannot save model to %s", (String)getParameter("modelPath"));
		LOGGER.throwing(Level.WARN, e);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#loadModel()
	 */
	@Override
	protected void loadModel() throws IOException {
		String currentTask = getCurrentTask();
		if(networkModel != null && (currentTask.equals(TASK_TEST) || currentTask.equals(TASK_TUNE))){
			LOGGER.info("Model already loaded, using loaded model.");
		} else {
			String modelPath = getParameter("modelPath");
			if(modelPath == null){
				throw LOGGER.throwing(Level.ERROR, new RuntimeException("Loading model requires --modelPath to be set."));
			}
			LOGGER.info("Reading model from %s...", modelPath);
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
			long startTime = System.nanoTime();
			try {
				networkModel = (NetworkModel)ois.readObject();
				instanceParser = (InstanceParser)ois.readObject();
			} catch (ClassNotFoundException e) {
				LOGGER.warn("Cannot load the model from %s", modelPath);
				throw new RuntimeException(LOGGER.throwing(Level.FATAL, e));
			} finally {
				ois.close();
			}
			long endTime = System.nanoTime();
			LOGGER.info("Reading model...Done in %.3fs", (endTime-startTime)/1.0e9);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#handleLoadModelError(java.lang.Exception)
	 */
	@Override
	protected void handleLoadModelError(Exception e) {
		LOGGER.error("Cannot load model from %s", (String)getParameter("modelPath"));
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
		//defined by user
		initTraining();
		
		//defined by user
		initAndSetInstanceParser();
		
		getInstancesForTraining();
		
		initGlobalNetworkParam();
		initAndSetNetworkCompiler();
		initAndSetFeatureManager();
		
		initNetworkModel();
		
	}

	protected Instance[] getInstancesForTraining(){
		if(hasParameter("trainInstances")){
			return getParameter("trainInstances");
		}
		if(!hasParameter("trainPath") || getParameter("trainPath") == null){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("The task %s requires --trainPath to be specified.", TASK_TRAIN)));
		}
		try {
			Instance[] trainInstances = instanceParser.buildInstances((String)getParameter("trainPath"));
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
		if(!hasParameter("trainPath") || getParameter("devPath") == null){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("The task %s requires --devPath to be specified.", TASK_TUNE)));
		}
		try {
			Instance[] devInstances = instanceParser.buildInstances((String)getParameter("devPath"));
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
		if(!hasParameter("trainPath") || getParameter("testPath") == null){
			throw LOGGER.throwing(Level.ERROR, new RuntimeException(String.format("The task %s requires --testPath to be specified.", TASK_TEST)));
		}
		try {
			Instance[] testInstances = instanceParser.buildInstances((String)getParameter("testPath"));
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
		return getParameter("trainInstances");
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#tune()
	 */
	@Override
	protected void train(Instance[] trainInstances) {
		try {
			this.resetTimer();
			long time = System.nanoTime();
			networkModel.train(trainInstances, getParameter("maxIter"));
			time = System.nanoTime() - time;
			this.timer = time;
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		LOGGER.info("Total training time: %.3fs\n", this.timer/1.0e9);
		if((boolean)getParameter("writeModelAsText")){
			String modelPath = getParameter("modelPath");
			String modelTextPath = modelPath+".txt";
			try{
				LOGGER.info("Writing model text into %s.txt...", modelTextPath);
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
				modelTextWriter.println("Labels:");
				List<String> labelsUsed = new ArrayList<String>(((DelimiterBasedInstanceParser)instanceParser).labels);
				Collections.sort(labelsUsed);
				modelTextWriter.println(labelsUsed);
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
							modelTextWriter.printf("\t\t%s %d %.17f\n", input, featureId, fm.getParam_G().getWeight(featureId));
						}
					}
				}
				modelTextWriter.close();
			} catch (IOException e){
				LOGGER.warn("Cannot write model text into %s.", modelTextPath);
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
		try {
			this.resetTimer();
			long time = System.nanoTime();
			Instance[] instanceWithPredictions = networkModel.decode(testInstances);
			time = System.nanoTime() - time;
			this.timer = time;
			setParameter("testInstances", instanceWithPredictions);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		LOGGER.info("Total testing time: %.3fs\n", this.timer/1.0e9);		
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#evaluationResult(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void evaluate(Instance[] instancesWithPrediction) {
		int corr = 0;
		int total = 0;
		for(Instance instance: instancesWithPrediction){
			@SuppressWarnings("unchecked")
			LinearInstance<String> linInstance = (LinearInstance<String>)instance;
			corr += linInstance.countNumCorrectlyPredicted();
			total += linInstance.size();
		}
		LOGGER.info("Correct/Total: %d/%d", corr, total);
		LOGGER.info("Accuracy: %.2f%%", 100.0*corr/total);
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#visualize(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void visualize(Instance[] instances) {
		String visualizerModelName = getParameter("linearModelClass")+"Viewer";
		try {
			@SuppressWarnings("unchecked")
			Class<VisualizationViewerEngine> visualizerModelClass = (Class<VisualizationViewerEngine>)Class.forName(visualizerModelName);
			networkModel.visualize(visualizerModelClass, instances);
		} catch (ClassNotFoundException e) {
			LOGGER.warn("Cannot automatically find viewer class for model name %s", (String)getParameter("linearModelClass"));
			LOGGER.throwing(Level.WARN, e);
		} catch (InterruptedException e) {
			LOGGER.info("Visualizer was interrupted.");
		}     
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#savePredictions()
	 */
	@Override
	protected void savePredictions() {
		if(!hasParameter("resultPath")){
			LOGGER.warn("Task savePredictions requires --resultPath to be specified.");
			return;
		}
		String resultPath = getParameter("resultPath");
		try{
			PrintWriter printer = new PrintWriter(new File(resultPath));
			LinearInstance<String>[] instances = getParameter("testInstances");
			for(LinearInstance<String> instance: instances){
				printer.println(instance.toString());
			}
			printer.close();
		} catch (FileNotFoundException e){
			LOGGER.warn("Cannot find file %s for storing prediction results.", resultPath);
		}
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#extractFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void extractFeatures(Instance[] instances) {
		// TODO
	}

	/* (non-Javadoc)
	 * @see com.statnlp.util.Pipeline#writeFeatures(com.statnlp.commons.types.Instance[])
	 */
	@Override
	protected void writeFeatures(Instance[] instances) {
		// TODO Auto-generated method stub
		
	}

}
