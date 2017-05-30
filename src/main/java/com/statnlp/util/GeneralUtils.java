package com.statnlp.util;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.FileAppender;
import org.apache.logging.log4j.core.config.AbstractConfiguration;
import org.apache.logging.log4j.core.config.AppenderRef;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.ConfigurationFactory;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.DefaultConfiguration;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.logging.log4j.core.config.Order;
import org.apache.logging.log4j.core.config.composite.CompositeConfiguration;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.xml.XmlConfiguration;
import org.apache.logging.log4j.core.config.xml.XmlConfigurationFactory;
import org.apache.logging.log4j.core.layout.PatternLayout;
import org.apache.logging.log4j.core.util.NetUtils;
import org.apache.logging.log4j.message.StringFormatterMessageFactory;
import org.apache.logging.log4j.util.LoaderUtil;
import org.apache.logging.log4j.util.PropertiesUtil;
import org.apache.logging.log4j.util.Strings;

/**
 * Class storing general utility methods.
 */
public class GeneralUtils {
	
	private static boolean configurationFactorySet = false;

	public static Logger createLogger(Class<?> clazz){
		if(!configurationFactorySet){
			ConfigurationFactory.setConfigurationFactory(new WithLogFileConfigurationFactory());
			configurationFactorySet = true;
		}
		return LogManager.getLogger(clazz, new StringFormatterMessageFactory());
	}

	public static List<String> sorted(Set<String> coll){
		List<String> result = new ArrayList<String>(coll);
		Collections.sort(result);
		return result;
	}
	
	public static void updateLogger(String logPath){
		WithLogFileConfiguration.updateLogger(logPath);
	}

	/**
	 * The factory class for configuration file for logging purpose<br>
	 * The various getConfiguration methods are part of an attempt to make the changes permanent 
	 * in the case of monitorInterval being used in the configuration file. But it doesn't work 
	 * as of now. Feeling that it should be close to the actual solution, I decided to leave them here.
	 * @author Aldrian Obaja (aldrianobaja.m@gmail.com)
	 */
	@Plugin(name = "WithLogFileConfigurationFactory", category = ConfigurationFactory.CATEGORY)
	@Order(10)
	public static class WithLogFileConfigurationFactory extends XmlConfigurationFactory {
		
		/**
		 * Valid file extensions.
		 */
		public static final String[] SUFFIXES = new String[] {".xml", "*"};
		
		private static final String ALL_TYPES = "*";

		/**
		 * Returns the file suffixes for configuration files.
		 * @return An array of File extensions.
		 */
		public String[] getSupportedTypes() {
			return SUFFIXES;
		}
		
		@Override
		public Configuration getConfiguration(LoggerContext loggerContext, ConfigurationSource source) {
			return new WithLogFileConfiguration(loggerContext, source);
		}

        /**
         * Default Factory Constructor.
         * @param name The configuration name.
         * @param configLocation The configuration location.
         * @return The Configuration.
         */
        @Override
        public Configuration getConfiguration(final LoggerContext loggerContext, final String name, final URI configLocation) {
            if (configLocation == null) {
                final String configLocationStr = this.substitutor.replace(PropertiesUtil.getProperties()
                        .getStringProperty(CONFIGURATION_FILE_PROPERTY));
                if (configLocationStr != null) {
                    final String[] sources = configLocationStr.split(",");
                    if (sources.length > 1) {
                        final List<AbstractConfiguration> configs = new ArrayList<>();
                        for (final String sourceLocation : sources) {
                            final Configuration config = getConfiguration(loggerContext, sourceLocation.trim());
                            if (config != null && config instanceof AbstractConfiguration) {
                                configs.add((AbstractConfiguration) config);
                            } else {
                                LOGGER.error("Failed to created configuration at {}", sourceLocation);
                                return null;
                            }
                        }
                        return new CompositeConfiguration(configs);
                    }
                    return getConfiguration(loggerContext, configLocationStr);
                }
            } else {
                // configLocation != null
            	final String configLocationStr = configLocation.toString();
            	final String[] types = getSupportedTypes();
            	if (types != null) {
            		for (final String type : types) {
            			if (type.equals(ALL_TYPES) || configLocationStr.endsWith(type)) {
            				final Configuration config = getConfiguration(loggerContext, name, configLocation);
            				if (config != null) {
            					return config;
            				}
            			}
            		}
            	}
            }

            Configuration config = getConfiguration(loggerContext, true, name);
            if (config == null) {
                config = getConfiguration(loggerContext, true, null);
                if (config == null) {
                    config = getConfiguration(loggerContext, false, name);
                    if (config == null) {
                        config = getConfiguration(loggerContext, false, null);
                    }
                }
            }
            if (config != null) {
                return config;
            }
            LOGGER.error("No log4j2 configuration file found. " +
                    "Using default configuration: logging only errors to the console. " +
                    "Set system property 'org.apache.logging.log4j.simplelog.StatusLogger.level'" +
                    " to TRACE to show Log4j2 internal initialization logging.");
            return new DefaultConfiguration();
        }

        private Configuration getConfiguration(final LoggerContext loggerContext, final String configLocationStr) {
            ConfigurationSource source = null;
            try {
                source = getInputFromUri(NetUtils.toURI(configLocationStr));
            } catch (final Exception ex) {
                // Ignore the error and try as a String.
                LOGGER.catching(Level.DEBUG, ex);
            }
            if (source == null) {
                final ClassLoader loader = LoaderUtil.getThreadContextClassLoader();
                source = getInputFromString(configLocationStr, loader);
            }
            if (source != null) {
            	final String[] types = getSupportedTypes();
            	if (types != null) {
            		for (final String type : types) {
            			if (type.equals(ALL_TYPES) || configLocationStr.endsWith(type)) {
            				final Configuration config = getConfiguration(loggerContext, source);
            				if (config != null) {
            					return config;
            				}
            			}
            		}
            	}
            }
            return null;
        }

        private Configuration getConfiguration(final LoggerContext loggerContext, final boolean isTest, final String name) {
            final boolean named = Strings.isNotEmpty(name);
            final ClassLoader loader = LoaderUtil.getThreadContextClassLoader();
            String configName;
            final String prefix = isTest ? TEST_PREFIX : DEFAULT_PREFIX;
            final String [] types = getSupportedTypes();
            if (types == null) {
            	return null;
            }

            for (final String suffix : types) {
            	if (suffix.equals(ALL_TYPES)) {
            		continue;
            	}
            	configName = named ? prefix + name + suffix : prefix + suffix;

            	final ConfigurationSource source = getInputFromResource(configName, loader);
            	if (source != null) {
            		return getConfiguration(loggerContext, source);
            	}
            }
            return null;
        }
	}

	/**
	 * The custom configuration class which appends the logs into an additional file.
	 * @author Aldrian Obaja (aldrianobaja.m@gmail.com)
	 */
	public static class WithLogFileConfiguration extends XmlConfiguration {
		private static String logPath = null;
		private static boolean changed = false;

		public WithLogFileConfiguration(final LoggerContext loggerContext, final ConfigurationSource configSource) {
			super(loggerContext, configSource);
		}
		
		@Override
		protected void doConfigure() {
			super.doConfigure();
			updateLogger();
		}

		@Override
	    public Configuration reconfigure() {
	        try {
	            final ConfigurationSource source = getConfigurationSource().resetInputStream();
	            if (source == null) {
	                return null;
	            }
	            final WithLogFileConfiguration config = new WithLogFileConfiguration(getLoggerContext(), source);
	            updateLogger();
	            return config;
	        } catch (final IOException ex) {
	            LOGGER.error("Cannot locate file {}", getConfigurationSource(), ex);
	        }
	        return null;
	    }

		private static void updateLogger(){
			if(!configurationFactorySet){
				ConfigurationFactory.setConfigurationFactory(new WithLogFileConfigurationFactory());
				configurationFactorySet = true;
				return;
			}
			if(logPath == null){
				return;
			}
			final LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
			final Configuration config = ctx.getConfiguration();
			PatternLayout layout = PatternLayout.newBuilder()
									.withPattern("%d{HH:mm:ss.SSS} [%t] %-5level %logger{36} - %msg%n")
									.withConfiguration(config)
									.build();
			FileAppender appender = ((org.apache.logging.log4j.core.util.Builder<FileAppender>)FileAppender.newBuilder()
									.withFileName(logPath)
									.withAppend(!changed) // If the logPath is not changed, just continue appending
									.withLocking(false)
									.withName("File")
									.withImmediateFlush(true)
									.withIgnoreExceptions(false)
									.withBufferedIo(true)
									.withLayout(layout)
									.withAdvertise(true)
									.setConfiguration(config))
									.build();
			appender.start();
			changed = false;
			config.addAppender(appender);
			AppenderRef ref = AppenderRef.createAppenderRef(appender.getName(), null, null);
			AppenderRef[] refs = new AppenderRef[] {ref};
			LoggerConfig loggerConfig = LoggerConfig.createLogger(false, Level.INFO, LogManager.ROOT_LOGGER_NAME,
					"true", refs, null, config, null );
			loggerConfig.addAppender(appender, null, null);
			config.addLogger(LogManager.ROOT_LOGGER_NAME, loggerConfig);
			ctx.updateLoggers(config);
			for(org.apache.logging.log4j.core.Logger logger: ctx.getLoggers()){
				logger.addAppender(appender);
			}
		}

		/**
		 * Update current loggers to add an additional appender to the specified log file
		 * @param logPath
		 */
		public static void updateLogger(String logPath){
			setLogPath(logPath);
			updateLogger();
		}

		private static void setLogPath(String logPath){
			WithLogFileConfiguration.logPath = logPath;
			WithLogFileConfiguration.changed = true;
		}
	}

}
