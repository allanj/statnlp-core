# -*- encoding: utf-8 -*-

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import nltk.data
import sys, random, codecs
import numpy as np
from pattern.en import pluralize, comparative, superlative, conjugate
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim
from pywsd.baseline import first_sense

random.seed(1)

stops = set(stopwords.words('english')+['I'])
# vmap = {"VB":"inf","VBP":"1sg","VBZ":"3sg","VBG":"part","VBD":"p","VBN":"ppart"}
exceptlist = {"saddle horse":"mount","base on balls":"pass","boundary line":None,'diethylstilbestrol':'des','new jersey':'jersey','last':'lowest','forted':'fort','mounted':'mount','scott':'scotts','rio de janeiro':'rio','riverbank':'riverside','tabun':'ga','extremum':'peak','district of columbia':'dc'}

testids={}
for ids in open("testids"):
    testids[int(ids)] = True

print "Processing en"
synlist = []
fout = open('en-syn.txt','w')
cnt = 0
for text in open("en.txt"):
    if cnt not in testids:
        synlist.append([])
        fout.write(text.lower().strip()+"\n")
        cnt += 1
        continue

    # Load a text file if required
    #text = "Pete ate a large cake. Sam has a big mouth."
    output = ""
    text = text.strip()

    # Get the list of words from the entire text
    words = word_tokenize(text)

    # Identify the parts of speech
    tagged = nltk.pos_tag(words)

    # Sense disambiguation
    #wsd = disambiguate(text, algorithm=maxsim, similarity_option='wup')

    synlist.append([])
    for i in range(0,len(words)):
        words[i] = words[i].lower()
        is_stop = words[i] in stops
        is_ok_pos= tagged[i][1].startswith('VB') or tagged[i][1] == 'NN' or tagged[i][1] == 'NNS' or tagged[i][1].startswith('JJ')
        if is_stop or not is_ok_pos:
            output += ' '+words[i]
        else:
            if tagged[i][1].startswith('VB'): pos = wordnet.VERB
            elif tagged[i][1].startswith('NN'): pos = wordnet.NOUN
            else: pos = wordnet.ADJ
            syns = wordnet.synsets(words[i],pos)
            syn = None
            if syns: syn = syns[0]
            #syn = wsd[i][1]

            if syn:
                lemmas = syn.lemma_names()
                for j in range(len(lemmas)):
                    lemmas[j] = lemmas[j].lower()
                if words[i] in lemmas:
                    first_lemma = lemmas[0].replace("_"," ")
                    random_lemma = random.choice(lemmas)
                    while len(random_lemma.split("_")) > 1:
                        random_lemma = random.choice(lemmas)
                    #random_lemma = lemmas[0]
                    #new_word = random_lemma.replace('_',' ')
                    new_word = random_lemma

                    if first_lemma not in exceptlist:
                        output += ' '+first_lemma
                    else:
                        output += ' '+words[i]
                    if new_word not in exceptlist and new_word != words[i]:
                        synlist[-1].append(syn)
                else:
                    output += ' '+words[i]
            else:
                output += ' '+words[i]
    fout.write(output.lower().strip()+'\n')
fout.close()

langs = ['cmn','deu','ell','fas','ind','swe','tha']
langs2d = {'cmn':'zh','deu':'de','ell':'el','fas':'fa','ind':'id','swe':'sv','tha':'th'}

def process(lang):
    lang2d = langs2d[lang]
    fout = codecs.open(lang2d+"-syn.txt",'w','utf-8')
    line_cnt = 0
    for text in codecs.open(lang2d+".txt",'r','utf-8'):
        if line_cnt not in testids:
            synlist.append([])
            fout.write(text.lower().strip()+"\n")
            line_cnt += 1
            continue

        text = text.strip()
        words = text.split()

        output = ""
        for i in range(0,len(words)):
            words[i] = words[i].lower()
            syns = wordnet.synsets(words[i],lang=lang)
            if syns:
                found = False
                for syn in syns:
                    if syn in synlist[line_cnt]:
                        lemmas = syn.lemma_names(lang)
                        for j in range(len(lemmas)):
                            lemmas[j] = lemmas[j].lower()
                        if words[i] in lemmas:
                            random_lemma = random.choice(lemmas)
                            while len(random_lemma.split("_")) > 1:
                                random_lemma = random.choice(lemmas)
                            #random_lemma = lemmas[0]
                            #new_word = random_lemma.replace('_',' ')
                            new_word = random_lemma
                            output += ' '+new_word
                        else:
                            output += ' '+words[i]
                        found = True
                        break
                if not found: output += ' '+words[i]
            else:
                output += ' '+words[i]
        line_cnt += 1
        fout.write(output.strip()+"\n")
    fout.close()

for lang in langs:
    print "Processing",lang
    process(lang)
