/**
 * 
 */
package com.statnlp.commons;

import java.io.PrintStream;

/**
 * Collection of static methods
 */
public class Utils {

	public static void print(String string, PrintStream... streams){
		if(streams.length == 0){
			streams = new PrintStream[]{System.out};
		}
		for(PrintStream stream: streams){
			stream.println(string);
		}
	}
}
