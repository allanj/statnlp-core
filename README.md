## StatNLP: Hypergraph-based Structured Prediction Framework


StatNLP structured prediction framework developed by [StatNLP team](http://www.statnlp.org/) provdies a way for NLP researchers to rapidly develop structured models including conditional random fields (CRF), structured perceptron, structured SVM, softmax-margin CRF as well as neural CRF with various inference strategies.

The theory behind is based on the unified view of structured prediction. Check out our tutorial in EMNLP 2017: [A Unified Framework for Structured Prediction: From Theory to Practice](http://emnlp2017.net/tutorials/day2/struct_pred.html). TThis framework is based on the concept of hypergraph, making it very general, which covers linear-chain graphical models -- such as HMM, Linear-chain CRFs, and semi-Markov CRFs, tree-based graphical models -- such as Tree CRFs for constituency parsing --, and many more.

Coupled with a generic training formulation based on the generalization of the inside-outside algorithm to acyclic hypergraphs, this framework supports the rapid prototyping and creation of novel graphical models, where users would just need to specify the graphical structure, and the framework will handle the training procedure. The neural component is integrated with [Torch 7](http://torch.ch/) package. 


### Existing Research 
A number of research papers([Jie and Lu, 2018](); [Zou and Lu, 2018](http://aclweb.org/anthology/P18-2107); [Li and Lu, 2018](http://aclweb.org/anthology/P18-2085); [Muis and Lu, 2017](http://www.statnlp.org/research/ie/emnlp2017-mention-separators.pdf); [Amoualian et al., 2017](http://www.aclweb.org/anthology/P17-1165); [Lim et al., 2017](http://www.statnlp.org/research/re/MalwareTextDB-1.0.pdf); [Jie et al., 2017](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14741/14133); [Li and Lu, 2017](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14931); [Susanto and Lu, 2017](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14843); [Lu et al., 2016](http://www.statnlp.org/research/ml/emnlp2016-720.pdf); [Muis and Lu, 2016a](http://www.statnlp.org/research/ie/emnlp2016-discontiguous-entities.pdf); [Muis and Lu, 2016b](http://aclweb.org/anthology/N/N16/N16-1085.pdf);  [Lu, 2015](http://www.aclweb.org/anthology/P15-2121); [Lu and Roth, 2015](http://www.aclweb.org/anthology/D/D15/D15-1102.pdf);[Lu, 2014](http://emnlp2014.org/papers/pdf/EMNLP2014137.pdf);) in references below have successfully use our framework to produce their novel model. 

### References
* Zhanming Jie and Wei Lu, _Dependency-based Hybrid Tree for Semantic Parsing_, EMNLP 2018.
* Yanyan Zou and Wei Lu, _Learning Cross-lingual Distributed Logical Representations for Semantic Parsing_, ACL 2018
* Hao Li and Wei Lu, _Learning with Structured Representations for Negation Scope Extraction_, ACL 2018
* Aldrian Obaja Muis and Wei Lu, _Labeling Gaps Between Words: Recognizing Overlapping Mentions with Mention Separators_, EMNLP 2017
* Hesam Amoualian et al., _Topical Coherence in LDA-based Models through Induced Segmentation_, ACL 2017
* Lim et al., _MalwareTextDB: A Database for Annotated Malware Articles_, ACL 2017
* Jie et al., _Efficient Dependency-guided Named Entity Recognition_, AAAI 2017
* Hao Li and Wei Lu., _Learning Latent Sentiment Scopes for Entity-Level Sentiment Analysis_, AAAI 2017
* Raymond Hendy Susanto and Wei Lu, _Semantic Parsing with Neural Hybrid Trees_, AAAI 2017
* Lu et al., _A General Regularization Framework for Domain Adaptation_, EMNLP 2016
* Aldrian Obaja Muis and Wei Lu, _Learning to Recognize Discontiguous Entities_, EMNLP 2016
* Aldrian Obaja Muis and Wei Lu, _Weak Semi-Markov CRFs for NP Chunking in Informal Text_, NAACL 2016
* Wei Lu and Dan Roth, _Joint Mention Extraction and Classification with Mention Hypergraphs_, EMNLP 2015
* Wei Lu, _Constrained Semantic Forests for Improved Discriminative Semantic Parsing_, ACL 2015
* Wei Lu, _Semantic Parsing with Relaxed Hybrid Trees_, EMNLP 2014


