package org.statnlp.example.benchmark_pytorch;

import java.io.BufferedReader;
import java.io.IOException;

import org.statnlp.commons.io.RAWF;
import org.statnlp.hypergraph.StringIndex;

public class BMUtils {

	public static void modifyWeight(double[] transition, int[][] featureRep, StringIndex str2idx, String weightFile) {
		double[][] readWeights = new double[5][5];
		BufferedReader br;
		try {
			br = RAWF.reader(weightFile);
			String line = null;
			int lineNum = 0;
			while((line = br.readLine()) != null) {
				String[] vals = line.trim().split("\\s+");
				for (int col = 0; col < vals.length; col++) {
					readWeights[col][lineNum] = Double.valueOf(vals[col]);
				}
				lineNum++;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		for (int w = 0; w < transition.length; w++) {
			int currIdx = label2idx(str2idx.get(featureRep[w][1]));
			int prevIdx = label2idx(str2idx.get(featureRep[w][2]));
			transition[w] = readWeights[prevIdx][currIdx];
		}
	}
	
	private static int label2idx(String label) {
		if(label.equals("B")) {
			return 0;
		} else if (label.equals("I")) {
			return 1;
		} else if (label.equals("O")) {
			return 2;
		} else if (label.equals("<START>")) {
			return 3;
		} else if (label.equals("<STOP>")) {
			return 4;
		} else {
			throw new RuntimeException("unknown label: " + label);
		}
	}
	
}
