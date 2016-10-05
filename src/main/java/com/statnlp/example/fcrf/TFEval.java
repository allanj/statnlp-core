package com.statnlp.example.fcrf;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import com.statnlp.commons.io.RAWF;
import com.statnlp.commons.types.Instance;
import com.statnlp.commons.types.Sentence;
import com.statnlp.example.fcrf.utils.DPConfig;

public class TFEval {

	
	/**
	 * 
	 * @param testInsts
	 * @param nerOut: word, true pos, true entity, pred entity
	 * @throws IOException
	 */
	public static void evalNER(Instance[] testInsts, String nerOut) throws IOException{
		PrintWriter pw = RAWF.writer(nerOut);
		for(int index=0;index<testInsts.length;index++){
			TFInstance eInst = (TFInstance)testInsts[index];
			ArrayList<String> predEntities = eInst.getEntityPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			Sentence sent = eInst.getInput();
			for(int i=0;i<sent.length();i++){
				pw.write(sent.get(i).getName()+" "+sent.get(i).getTag()+" "+trueEntities.get(i)+" "+predEntities.get(i)+"\n");
			}
			pw.write("\n");
		}
		pw.close();
		evalNER(nerOut);
	}
	
	
	private static void evalNER(String outputFile) throws IOException{
		try{
			System.err.println("perl data/semeval10t1/conlleval.pl < "+outputFile);
			ProcessBuilder pb = null;
			if(DPConfig.windows){
				pb = new ProcessBuilder("D:/Perl64/bin/perl","E:/Framework/data/semeval10t1/conlleval.pl"); 
			}else{
				pb = new ProcessBuilder("data/conlleval.pl"); 
			}
			pb.redirectInput(new File(outputFile));
			pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
			pb.redirectError(ProcessBuilder.Redirect.INHERIT);
			pb.start();
		}catch(IOException ioe){
			ioe.printStackTrace();
		}
	}
	
	public static void evalPOS(Instance[] testInsts, String posOut) throws IOException{
		PrintWriter pw = RAWF.writer(posOut);
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			TFInstance eInst = (TFInstance)testInsts[index];
			ArrayList<String> tPred = eInst.getTagPredictons();
			Sentence sent = eInst.getInput();
			for(int i=0;i<sent.length();i++){
				if(sent.get(i).getTag().equals(tPred.get(i)))
					corr++;
				total++;
				pw.write(sent.get(i).getName()+" "+sent.get(i).getTag()+" "+tPred.get(i)+"\n");
			}
			pw.write("\n");
		}
		System.out.println("[POS Result]:"+ corr*1.0/total);
		pw.close();
	}
	
	public static void evalSingleE(Instance[] testInsts) throws IOException{
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			TFInstance eInst = (TFInstance)testInsts[index];
			ArrayList<String> tPred = eInst.getEntityPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			for(int i=0;i<tPred.size();i++){
				if(tPred.get(i).equals(trueEntities.get(i)))
					corr++;
				total++;
			}
		}
		System.out.println("[NE notion Result]:"+ corr*1.0/total);
	}
	
	public static void evalSingleJoint(Instance[] testInsts) throws IOException{
		int corr = 0;
		int total = 0;
		for(int index=0;index<testInsts.length;index++){
			TFInstance eInst = (TFInstance)testInsts[index];
			ArrayList<String> ePred = eInst.getEntityPredictons();
			ArrayList<String> trueEntities = eInst.getOutput();
			ArrayList<String> tPred = eInst.getTagPredictons();
			Sentence sent = eInst.getInput();
			for(int i=0;i<ePred.size();i++){
				if(ePred.get(i).equals(trueEntities.get(i)) && sent.get(i).getTag().equals(tPred.get(i)))
					corr++;
				total++;
			}
			
		}
		System.out.println("[joint notion Result]:"+ corr*1.0/total);
	}
	
}
