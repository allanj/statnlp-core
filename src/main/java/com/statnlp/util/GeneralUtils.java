/**
 * 
 */
package com.statnlp.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.message.StringFormatterMessageFactory;

/**
 * Class storing general utility methods.
 */
public class GeneralUtils {
	
	public static Logger createLogger(Class<?> clazz){
		return LogManager.getLogger(clazz, new StringFormatterMessageFactory());
	}
	
	public static List<String> sorted(Set<String> coll){
		List<String> result = new ArrayList<String>(coll);
		Collections.sort(result);
		return result;
	}

}
