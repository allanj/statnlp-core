/**
 * 
 */
package com.statnlp.util;

import com.statnlp.commons.ml.opt.OptimizerFactory;
import com.statnlp.example.TreeCRFMain;
import com.statnlp.example.tree_crf.TreeCRFFeatureManager;
import com.statnlp.example.tree_crf.TreeCRFNetworkCompiler;
import com.statnlp.hybridnetworks.NetworkConfig.ModelType;
import com.statnlp.hybridnetworks.NetworkConfig.StoppingCriteria;
import com.statnlp.util.instance_parser.TreebankInstanceParser;

/**
 * To test the implementation of {@link GenericPipeline} on TreeCRF implementation<br>
 * This showcases how to use the GenericPipeline with custom implementation.
 */
public class GenericPipelineTreeCRFTest {
	
	private static GenericPipeline pipeline;
	
	public static void main(String[] args) throws Exception {
		final boolean runFromArgs = false;
		if(runFromArgs){
			runFromArgs();
		} else {
			runFromCode();
		}
	}
	
	private static final void runFromArgs(){
		Runner.run(new String[]{
			    "--instanceParserClass", "com.statnlp.util.instance_parser.TreebankInstanceParser",
			    "--networkCompilerClass", "com.statnlp.example.tree_crf.TreeCRFNetworkCompiler",
			    "--featureManagerClass", "com.statnlp.example.tree_crf.TreeCRFFeatureManager",
			    "--evaluateCallback", "com.statnlp.example.tree_crf.TreeCRFNetworkCompiler::evaluate",
			    "--trainPath", "data/ptb-binary.train",
			    "--testPath", "data/ptb-binary.test",
			    "--modelPath", "tree_crf.model",
			    "--logPath", "tree_crf.log",
			    "--attemptMemorySaving",
			    "--numThreads", "4",
			    "--l2", "0.001",
			    "--useBatchTraining",
			    "--batchSize", "2",
			    "--useGD",
			    "--stoppingCriteria", "MAX_ITERATION_REACHED",
			    "--modelType", "SSVM",
			    "--maxIter", "200",
			    "--nodeMismatchCost", "1.0",
			    "--edgeMismatchCost", "0.0",
			    "--maxSentenceLength=50",
			    "train", "test", "evaluate",
		});
	}
	
	private static final void runFromCode(){
		pipeline = new GenericPipeline()
				.withTrainPath("data/ptb-binary.train")						// Specify the training data
				.withTestPath("data/ptb-binary.test")						// Specify the test data
				.withModelPath("test.model")								// Specify where to save the model (if not specified no model will be written)
				.withLogPath("test.log")									// Specify the log file
				.withAttemptMemorySaving(true)								// Save memory and time
				.withInstanceParser(TreebankInstanceParser.class)			// Specify the instance parser (the one responsible to read the data)
				.withFeatureManager(TreeCRFFeatureManager.class)			// Specify the feature manager
				.withNetworkCompiler(TreeCRFNetworkCompiler.class)			// Specify the network compiler
				.withEvaluateCallback(TreeCRFMain::evaluate)				// Specify the evaluation function
				.addTask("train")
				.addTasks("test", "evaluate")
				.withL2(0.0001)
				.withModelType(ModelType.SSVM)
				.withEdgeMismatchCost(0.0)
				.withNodeMismatchCost(1.0)
				.withOptimizerFactory(OptimizerFactory.getGradientDescentFactoryUsingAdaGrad(0.15))
				.withUseBatchTraining(true)
				.withBatchSize(1)
				.withMaxIter(800)
				.withStoppingCriteria(StoppingCriteria.MAX_ITERATION_REACHED)
				;
		pipeline.execute();
	}

}
