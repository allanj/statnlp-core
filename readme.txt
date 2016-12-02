The StatNLP Semantic Parser Version 0.2 (Relaxed Hybrid Trees, Hybrid Trees, Constrained Semantic Forests).

To run the code, simply issue the following command:

java -cp bin -Xmx125g com.statnlp.sp.main.SemTextExperimenter_Discriminative <NUMBER_OF_THREAD> <LANGUAGE>

For example:
java -cp bin -Xmx125g com.statnlp.sp.main.SemTextExperimenter_Discriminative 16 en

Make sure you have installed SICSTUS prolog in your system, and edit the path to prolog in the file "path_to_prolog". Also change edit the file "path_to_eval" to reflect the absolute path to the eval folder.

Note that this version is not yet optimized. It requires substantial RAM to run the code. We are in the middle of working on optimizing the code.

Wei Lu, Singapore University of Technology and Design. Please drop an email to luwei@sutd.edu.sg if you have queries.
