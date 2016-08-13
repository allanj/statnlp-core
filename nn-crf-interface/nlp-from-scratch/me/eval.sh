evalscript=/home/raymondhs/Workspace/semantic/sp+nn/statnlp-core/data/semeval10t1/conlleval.pl
python forEval.py ../data/conll2003/eng.testb ner_results_embedding.txt > output/ner.conll
python forEval.py ../data/conll2003/eng.testb ../nlpfromscratch/max_scores.txt > output/manual.conll
$evalscript < output/ner.conll
$evalscript < output/manual.conll
