/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.statnlp.hybridnetworks;

import static com.statnlp.commons.Utils.print;

import java.io.PrintStream;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.NetworkConfig.InferenceType;
import com.statnlp.neural.NeuralNetworkFeatureValueProvider;
import com.statnlp.ui.visualize.type.VisualizationViewerEngine;
import com.statnlp.ui.visualize.type.VisualizerFrame;

public abstract class NetworkModel implements Serializable{
	
	private static final Random RANDOM = new Random(NetworkConfig.RANDOM_BATCH_SEED);

	private static final long serialVersionUID = 8695006398137564299L;
	
	//the global feature manager.
	protected FeatureManager _fm;
	//the builder
	protected NetworkCompiler _compiler;
	//the list of instances.
	protected transient Instance[] _allInstances;
	protected transient Network[] unlabeledNetworkByInstanceId;
	protected transient Network[] labeledNetworkByInstanceId;
	//the number of threads.
	protected transient int _numThreads = NetworkConfig.NUM_THREADS;
	//the local learners.
	private transient LocalNetworkLearnerThread[] _learners;
	//the local decoder.
	private transient LocalNetworkDecoderThread[] _decoders;
	//neuralCRF/SSVM socket controller
//	private NNCRFGlobalNetworkParam nnController;
	private transient PrintStream[] outstreams = new PrintStream[]{System.out};
	
	public NetworkModel(FeatureManager fm, NetworkCompiler compiler, PrintStream... outstreams){
		this._fm = fm;
		this._numThreads = NetworkConfig.NUM_THREADS;
		this._compiler = compiler;
		if(outstreams == null){
			outstreams = new PrintStream[0];
		}
		this.outstreams = new PrintStream[outstreams.length+1];
		this.outstreams[0] = System.out;
		for(int i=0; i<outstreams.length; i++){
			this.outstreams[i+1] = outstreams[i];
		}
	}
	
	public int getNumThreads(){
		return this._numThreads;
	}
	
	public Network getLabeledNetwork(int instanceId){
		return labeledNetworkByInstanceId[instanceId-1];
	}
	
	public Network getUnlabeledNetwork(int instanceId){
		return unlabeledNetworkByInstanceId[instanceId-1];
	}
	
	public Instance[] getInstances(){
		return _allInstances;
	}
	
	protected abstract Instance[][] splitInstancesForTrain();
	
	public Instance[][] splitInstancesForTest() {
		
		System.err.println("#instances="+this._allInstances.length);
		
		Instance[][] insts = new Instance[this._numThreads][];

		ArrayList<ArrayList<Instance>> insts_list = new ArrayList<ArrayList<Instance>>();
		int threadId;
		for(threadId = 0; threadId<this._numThreads; threadId++){
			insts_list.add(new ArrayList<Instance>());
		}
		
		threadId = 0;
		for(int k = 0; k<this._allInstances.length; k++){
			Instance inst = this._allInstances[k];
			insts_list.get(threadId).add(inst);
			threadId = (threadId+1)%this._numThreads;
		}
		
		for(threadId = 0; threadId<this._numThreads; threadId++){
			int size = insts_list.get(threadId).size();
			insts[threadId] = new Instance[size];
			for(int i = 0; i < size; i++){
				Instance inst = insts_list.get(threadId).get(i);
				insts[threadId][i] = inst;
			}
			print("Thread "+threadId+" has "+insts[threadId].length+" instances.", outstreams);
		}
		
		return insts;
	}

	/**
	 * Visualize the trained instances using the specified viewer engine.<br>
	 * Note that this assumes that the training is done.
	 * @param clazz The class of the viewer engine to be used to visualize the instances.
	 * @throws InterruptedException If there are interruptions during the compilation process of the instances in multi-threaded setting.
	 * @throws IllegalStateException If train method has not been called. This means there are no instances to be visualized.
	 * @see {@link #visualize(Class, Instance[])}
	 * @see {@link #visualize(Class, Instance[], int)}
	 */
	public void visualize() throws InterruptedException, IllegalStateException {
		visualize(VisualizationViewerEngine.class);
	}
	
	/**
	 * Visualize the trained instances using the specified viewer engine.<br>
	 * Note that this assumes that the training is done.
	 * @param clazz The class of the viewer engine to be used to visualize the instances.
	 * @throws InterruptedException If there are interruptions during the compilation process of the instances in multi-threaded setting.
	 * @throws IllegalStateException If train method has not been called. This means there are no instances to be visualized.
	 * @throws IllegalArgumentException If the viewer engine specified does not implement the correct constructor.
	 * 									The viewer engine needs to implement the constructor with signature (NetworkCompiler, FeatureManager)
	 * @see {@link #visualize(Class, Instance[])}
	 * @see {@link #visualize(Class, Instance[], int)}
	 */
	public void visualize(Class<? extends VisualizationViewerEngine> clazz) throws InterruptedException, IllegalArgumentException, IllegalStateException{
		if(unlabeledNetworkByInstanceId == null){
			throw new IllegalStateException("Please specify the instances to be visualized.");
		}
		visualize(clazz, null, _allInstances.length);
	}
	

	/**
	 * Visualize the instances using the specified viewer engine.
	 * @param clazz The class of the viewer engine to be used to visualize the instances.
	 * @param allInstances The instances to be visualized
	 * @throws IllegalArgumentException If the viewer engine specified does not implement the correct constructor.
	 * 									The viewer engine needs to implement the constructor with signature (NetworkCompiler, FeatureManager)
	 */
	public void visualize(Class<? extends VisualizationViewerEngine> clazz, Instance[] allInstances) throws InterruptedException{
		visualize(clazz, allInstances, allInstances.length);
	}
	
	/**
	 * Visualize the instances using the specified viewer engine.
	 * @param clazz The class of the viewer engine to be used to visualize the instances.
	 * @param allInstances The instances to be visualized
	 * @param trainLength The number of instances to be visualized. This can be less than the number of instances in allInstances
	 * @throws InterruptedException If there are interruptions during the compilation process of the instances in multi-threaded setting.
	 * @throws IllegalArgumentException If the viewer engine specified does not implement the correct constructor.
	 * 									The viewer engine needs to implement the constructor with signature (NetworkCompiler, FeatureManager)
	 */
	public void visualize(Class<? extends VisualizationViewerEngine> clazz, Instance[] allInstances, int trainLength) throws InterruptedException, IllegalArgumentException {
		if(allInstances != null){
			System.err.print("Compiling networks...");
			long start = System.nanoTime();
			preCompileNetworks(prepareInstanceForCompilation(allInstances, trainLength));
			long end = System.nanoTime();
			System.err.printf("Done in %.3fs\n", (end-start)/1.0e9);
		}
		try {
			new VisualizerFrame(this, clazz.getConstructor(NetworkCompiler.class, FeatureManager.class).newInstance(_compiler, _fm));
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
				| NoSuchMethodException | SecurityException e) {
			throw new IllegalArgumentException("The viewer class "+clazz.getName()+" must implement the constructor with signature (NetworkCompiler, FeatureManager)");
		}
	}
	
	/**
	 * Starts training using the instances given for up to specified number of iterations.
	 * @param allInstances The instances on which this model should be trained.
	 * @param maxNumIterations The maximum number of iterations the training should go.
	 * @throws InterruptedException If there are interruptions during multi-threaded training.
	 */
	public void train(Instance[] allInstances, int maxNumIterations) throws InterruptedException{
		train(allInstances, allInstances.length, maxNumIterations);
	}
	
	/**
	 * Starts training using the instances given for up to specified number of iterations.
	 * @param allInstances The instances on which this model should be trained.
	 * @param trainLength The number of training instances used. This can be less than the number of instances in allInstances.
	 * @param maxNumIterations The maximum number of iterations the training should go.
	 * @throws InterruptedException If there are interruptions during multi-threaded training.
	 */
	public void train(Instance[] allInstances, int trainLength, int maxNumIterations) throws InterruptedException{
		Instance[][] insts = prepareInstanceForCompilation(allInstances, trainLength);
		ArrayList<Integer> instIds = new ArrayList<Integer>();
		for(int i=0; i<trainLength; i++){
			instIds.add(i+1);
		}
		/*
		 * Pre-compile the networks
		 * In mean-field, we need to pre-compile because we need the unlabeled network
		 * information in feature extraction process for the labeled network.
		 */
		NetworkConfig.PRE_COMPILE_NETWORKS = NetworkConfig.INFERENCE == InferenceType.MEAN_FIELD ? true : false;
		if(NetworkConfig.PRE_COMPILE_NETWORKS){
			preCompileNetworks(insts);
		}
		boolean keepExistingThreads = NetworkConfig.PRE_COMPILE_NETWORKS ? true : false;
		// The first touch
		touch(insts, keepExistingThreads);
		
		for(int threadId=0; threadId<this._numThreads; threadId++){
			if(NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY){
				// We extract features only from labeled instance in the first touch, so we don't know what
				// features are present in each thread. So copy all features to each thread.
				// Since we extract only from labeled instances, the feature index will be smaller
				this._fm.addIntoLocalFeatures(this._learners[threadId].getLocalNetworkParam()._globalFeature2LocalFeature);
			}
			this._learners[threadId].getLocalNetworkParam().finalizeIt();
		}

		//complete the type2int map. only in generative model
		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
			this._fm.completeType2Int(); 
		}
		
		//finalize the features.
		this._fm.getParam_G().lockIt();
//		nnController = this._fm.getParam_G()._nnController;
		
		if(NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY && NetworkConfig.CACHE_FEATURES_DURING_TRAINING){
			touch(insts, true); // Touch again to cache the features, both in labeled and unlabeled
		}
		
		if(NetworkConfig.CACHE_FEATURES_DURING_TRAINING){
			for(int threadId=0; threadId<this._numThreads; threadId++){
				// This was previously in each LocalNetworkLearnerThread finalizeIt, but moved here since
				// when we call finalizeIt above, it should not delete this variable first, because we were
				// using it in the second touch.
				this._learners[threadId].getLocalNetworkParam()._globalFeature2LocalFeature = null;
			}
		}
		
		ExecutorService pool = Executors.newFixedThreadPool(this._numThreads);
		List<Callable<Void>> callables = Arrays.asList((Callable<Void>[])this._learners);
		
		int multiplier = 1; // By default, print the negative of the objective
		if(!NetworkConfig.MODEL_TYPE.USE_SOFTMAX){
			// Print the objective if not using softmax 
			multiplier = -1;
		}

		HashSet<Integer> batchInstIds = new HashSet<Integer>();
		double obj_old = Double.NEGATIVE_INFINITY;
		//run the EM-style algorithm now...
		long startTime = System.currentTimeMillis();
		try{
			int batchId = 0;
			int epochNum = 0;
			double epochObj = 0.0;
			for(int it = 0; it<=maxNumIterations; it++){
				//at each iteration, shuffle the inst ids. and reset the set, which is already in the learner thread
				if(NetworkConfig.USE_BATCH_TRAINING){
					batchInstIds.clear();
					if(NetworkConfig.RANDOM_BATCH || batchId == 0) {
						Collections.shuffle(instIds, RANDOM);
					}
					int size = NetworkConfig.BATCH_SIZE >= this._allInstances.length ? this._allInstances.length:NetworkConfig.BATCH_SIZE; 
					for(int iid = 0; iid<size; iid++){
						int offset = NetworkConfig.BATCH_SIZE*batchId;
						batchInstIds.add(instIds.get(iid+offset));
					}
					if(!NetworkConfig.RANDOM_BATCH) {
						batchId++;
						int offset = NetworkConfig.BATCH_SIZE*batchId;
						if(size+offset > instIds.size()) {
							batchId = 0;
							// this means one epoch
							print(String.format("Epoch %d: Obj=%-18.12f", epochNum++, epochObj/instIds.size()), outstreams);
							epochObj = 0.0;
						}
					}
				}
				for(LocalNetworkLearnerThread learner: this._learners){
					learner.setIterationNumber(it);
					if(NetworkConfig.USE_BATCH_TRAINING) learner.setInstanceIdSet(batchInstIds);
					else learner.setTrainInstanceIdSet(new HashSet<Integer>(instIds));
				}
				long time = System.currentTimeMillis();
				
				// Feature value provider's ``forward''
				this._fm.getParam_G().computeContinousScores();
				this._fm.getParam_G().resetGradContinuous();
				
				List<Future<Void>> results = pool.invokeAll(callables);
				for(Future<Void> result: results){
					try{
						result.get(); // To ensure any exception is thrown
					} catch (ExecutionException e){
						throw new RuntimeException(e);
					}
				}
				
				boolean done = true;
				boolean lastIter = (it == maxNumIterations);
				if(lastIter){
					done = this._fm.update(true);
				} else {
					done = this._fm.update();
				}
				time = System.currentTimeMillis() - time;
				double obj = this._fm.getParam_G().getObj_old();
				epochObj += obj;
				print(String.format("Iteration %d: Obj=%-18.12f Time=%.3fs %.12f Total time: %.3fs", it, multiplier*obj, time/1000.0, obj/obj_old, (System.currentTimeMillis()-startTime)/1000.0), outstreams);
				if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE && it>1 && obj<obj_old && Math.abs(obj-obj_old)>1E-5){
					throw new RuntimeException("Error:\n"+obj_old+"\n>\n"+obj);
				}
				obj_old = obj;
				if (lastIter || done) {
					this._fm.getParam_G().computeContinousScores();
				}
				if(lastIter){
					print("Training completes. The specified number of iterations ("+it+") has passed.", outstreams);
					break;
				}
				if(done){
					print("Training completes. No significant progress (<objtol) after "+it+" iterations.", outstreams);
					break;
				}
			}
		} finally {
			pool.shutdown();
		}
	}

	private Instance[][] prepareInstanceForCompilation(Instance[] allInstances, int trainLength) {
		this._numThreads = NetworkConfig.NUM_THREADS;
		
		this._allInstances = allInstances;
		for(int k = 0; k<this._allInstances.length; k++){
			this._allInstances[k].setInstanceId(k+1);
		}
		this._fm.getParam_G().setInstsNum(this._allInstances.length);
		
		//create the threads.
		this._learners = new LocalNetworkLearnerThread[this._numThreads];
		
		return this.splitInstancesForTrain();
	}

	private void preCompileNetworks(Instance[][] insts) throws InterruptedException{
		for(int threadId = 0; threadId < this._numThreads; threadId++){
			this._learners[threadId] = new LocalNetworkLearnerThread(threadId, this._fm, insts[threadId], this._compiler, -1);
			this._learners[threadId].setPrecompile();
			this._learners[threadId].start();
		}
		for(int threadId = 0; threadId < this._numThreads; threadId++){
			this._learners[threadId].join();
			this._learners[threadId].unsetPrecompile();
		}	
		saveCompiledNetworks(insts);
		System.err.println("Finish precompile the networks.");
	}
	
	private void touch(Instance[][] insts, boolean keepExisting) throws InterruptedException {
		if(!NetworkConfig.PARALLEL_FEATURE_EXTRACTION || NetworkConfig.NUM_THREADS == 1){
			for(int threadId = 0; threadId<this._numThreads; threadId++){
				if(!keepExisting){
					this._learners[threadId] = new LocalNetworkLearnerThread(threadId, this._fm, insts[threadId], this._compiler, 0);
				} else {
					this._learners[threadId] = this._learners[threadId].copyThread();
				}
				this._learners[threadId].touch();
				System.err.println("Okay..thread "+threadId+" touched.");
			}
		} else {
			for(int threadId = 0; threadId < this._numThreads; threadId++){
				if(!keepExisting){
					this._learners[threadId] = new LocalNetworkLearnerThread(threadId, this._fm, insts[threadId], this._compiler, -1);
				} else {
					this._learners[threadId] = this._learners[threadId].copyThread();
				}
				this._learners[threadId].setTouch();
				this._learners[threadId].start();
			}
			for(int threadId = 0; threadId < this._numThreads; threadId++){
				this._learners[threadId].join();
				this._learners[threadId].setUnTouch();
			}
			
			if(NetworkConfig.PRE_COMPILE_NETWORKS || !keepExisting){
				//this one is because in the first touch, we don't have the exisiting threads if not precompile network. 
				//So we merge in first feature extraction.
				//If precompile network, we keep exisiting threads. But we still need to merge features because
				//it is still the first touch. That's why we have the "OR" operation here.
				this._fm.mergeSubFeaturesToGlobalFeatures();
			}
		}
		saveCompiledNetworks(insts);
	}

	private void saveCompiledNetworks(Instance[][] insts) {
		if(labeledNetworkByInstanceId == null || unlabeledNetworkByInstanceId == null){
			labeledNetworkByInstanceId = new Network[this._allInstances.length];
			unlabeledNetworkByInstanceId = new Network[this._allInstances.length];
			Network[] arr;
			for(int threadId=0; threadId < insts.length; threadId++){
				LocalNetworkLearnerThread learner = this._learners[threadId];
				for(int networkId=0; networkId < insts[threadId].length; networkId++){
					Instance instance = insts[threadId][networkId];
					int instanceId = instance.getInstanceId();
					if(instanceId < 0){
						arr = unlabeledNetworkByInstanceId;
						instanceId = -instanceId;
					} else {
						arr = labeledNetworkByInstanceId;
					}
					instanceId -= 1;
					arr[instanceId] = learner.getNetwork(networkId);
				}
			}
			if(unlabeledNetworkByInstanceId[0] == null){
				arr = labeledNetworkByInstanceId;
				labeledNetworkByInstanceId = unlabeledNetworkByInstanceId;
				unlabeledNetworkByInstanceId = arr;
			}
		}
	}
	
	public Instance[] decode(Instance[] allInstances) throws InterruptedException {
		return decode(allInstances, false);
	}
	
	public Instance[] decode(Instance[] allInstances, boolean cacheFeatures) throws InterruptedException{
		return decode(allInstances, cacheFeatures, 1);
	}
	
	public Instance[] decode(Instance[] allInstances, int numPredictionsGenerated) throws InterruptedException {
		return decode(allInstances, false, numPredictionsGenerated);
	}
	
	public Instance[] decode(Instance[] allInstances, boolean cacheFeatures, int numPredictionsGenerated) throws InterruptedException{
		
//		if(NetworkConfig.TRAIN_MODE_IS_GENERATIVE){
//			this._fm.getParam_G().expandFeaturesForGenerativeModelDuringTesting();
//		}
		
		this._numThreads = NetworkConfig.NUM_THREADS;
		System.err.println("#threads:"+this._numThreads);
		
		Instance[] results = new Instance[allInstances.length];
		
		//all the instances.
		this._allInstances = allInstances;
		
		//create the threads.
		if(this._decoders == null || !cacheFeatures){
			this._decoders = new LocalNetworkDecoderThread[this._numThreads];
		}
		
		Instance[][] insts = this.splitInstancesForTest();
		
		//distribute the works into different threads.
		for(int threadId = 0; threadId<this._numThreads; threadId++){
			if(cacheFeatures){
				if(this._decoders[threadId] != null){
					this._decoders[threadId] = new LocalNetworkDecoderThread(threadId, this._fm, insts[threadId], this._compiler, this._decoders[threadId].getParam(), true, numPredictionsGenerated);
				} else {
					this._decoders[threadId] = new LocalNetworkDecoderThread(threadId, this._fm, insts[threadId], this._compiler, true, numPredictionsGenerated);
				}
			} else {
				this._decoders[threadId] = new LocalNetworkDecoderThread(threadId, this._fm, insts[threadId], this._compiler, false, numPredictionsGenerated);
			}
		}
		
		if (NetworkConfig.FEATURE_TOUCH_TEST) {
			System.err.println("Touching test set.");
			
			for(int threadId = 0; threadId<this._numThreads; threadId++){
				this._decoders[threadId].setTouch();
				this._decoders[threadId].start();
			}
			for(int threadId = 0; threadId<this._numThreads; threadId++){
				this._decoders[threadId].join();
				this._decoders[threadId].setUnTouch();
			}
		}
		
		System.err.println("Okay. Decoding started.");
		
		long time = System.currentTimeMillis();
		
		this._fm.getParam_G().initializeProviderForDecoding();
		this._fm.getParam_G().computeContinousScores();
		
		for(int threadId = 0; threadId<this._numThreads; threadId++){
			this._decoders[threadId] = this._decoders[threadId].copyThread(this._fm);
			this._decoders[threadId].start();
		}
		for(int threadId = 0; threadId<this._numThreads; threadId++){
			this._decoders[threadId].join();
		}
		
		System.err.println("Okay. Decoding done.");
		time = System.currentTimeMillis() - time;
		System.err.println("Overall decoding time = "+ time/1000.0 +" secs.");
		
		int k = 0;
		for(int threadId = 0; threadId<this._numThreads; threadId++){
			Instance[] outputs = this._decoders[threadId].getOutputs();
			for(Instance output : outputs){
				results[k++] = output;
			}
		}
		
		Arrays.sort(results, Comparator.comparing(Instance::getInstanceId));
		
		return results;
	}
	
}
