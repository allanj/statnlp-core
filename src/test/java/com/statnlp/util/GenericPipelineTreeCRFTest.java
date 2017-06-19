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
	
	public static void main(String[] args){
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
