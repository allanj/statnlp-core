package com.statnlp.util.instance_parser;

import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.Map;

import com.statnlp.commons.types.Instance;
import com.statnlp.util.Pipeline;

/**
 * This InstanceParser class is used to parse training data and arguments and build instance and parameters for network compiler and feature manager
 * Users can override the {@link #buildInstances()} function to customize its own method for a specific algorithm
 * @author Li Hao
 *
 */
public abstract class InstanceParser implements Serializable{
	
	private static final long serialVersionUID = 5352091663516161611L;
	
	protected transient Pipeline pipeline;
	/**
	 * The parameters that might be associated with this parser.<br>
	 */
	private Map<String, Object> parameters;

	public InstanceParser(Pipeline pipeline) {
		this.pipeline = pipeline;
		this.parameters = pipeline.parameters.getAttrs();
	}
	
	public abstract Instance[] buildInstances(String... sources) throws FileNotFoundException;
	
	public void setParameter(String key, Object value){
		parameters.put(key, value);
	}
	
	@SuppressWarnings("unchecked")
	public <T> T getParameter(String key){
		return (T)parameters.get(key);
	}
	
}
