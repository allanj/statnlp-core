package com.statnlp.example.linear_ne;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.statnlp.commons.io.RAWF;
import com.statnlp.commons.types.Sentence;
import com.statnlp.commons.types.WordToken;

public class EReader {

	public static List<ECRFInstance> readData(String path, boolean setLabel, int number) throws IOException{
		BufferedReader br = RAWF.reader(path);
		String line = null;
		List<ECRFInstance> insts = new ArrayList<ECRFInstance>();
		int index =1;
		ArrayList<WordToken> words = new ArrayList<WordToken>();
		ArrayList<String> es = new ArrayList<String>();
		String prevLine = null;
		while((line = br.readLine())!=null){
			if(line.startsWith("-DOCSTART-")) { prevLine = "-DOCSTART-"; continue;}
			if(line.equals("") && !prevLine.equals("-DOCSTART-")){
				WordToken[] wordsArr = new WordToken[words.size()];
				words.toArray(wordsArr);
				Sentence sent = new Sentence(wordsArr);
				ECRFInstance inst = new ECRFInstance(index++,1.0,sent);
				inst.entities = es;
				if(setLabel) inst.setLabeled(); else inst.setUnlabeled();
				insts.add(inst);
				words = new ArrayList<WordToken>();
				es = new ArrayList<String>();
				prevLine = "";
				if(number!=-1 && insts.size()==number) break;
				continue;
			}
			if(line.equals("") && prevLine.equals("-DOCSTART-")){
				prevLine = ""; continue;
			}
			String[] values = line.split(" ");
			String entity = values[3];
			Entity.get(entity);
			words.add(new WordToken(values[1],values[2],-1,entity));
			es.add(entity);
			prevLine = line;
		}
		br.close();
		List<ECRFInstance> myInsts = insts;
		String type = setLabel? "Training":"Testing";
		System.err.println(type+" instance, total:"+ myInsts.size()+" Instance. ");
		return myInsts;
	}

	
}
