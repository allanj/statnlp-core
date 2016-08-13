import sys

words = []
caps = []
labels = []

labelSet = {}
#labelFile = "/Users/nlp/Documents/workspace/semantic/statnlp-core/nn-crf-interface/nlp-from-scratch/senna-torch/senna/hash/ner.lst"
labelFile = "../senna-torch/senna/hash/ner9.lst"
numLabel = 9
for line in open(labelFile):
    if len(labelSet) < numLabel:
        labelSet[line.strip()] = len(labelSet)+1

def apply_offset(lst, idx, offset, pad):
    if idx+offset < 0 or idx+offset > len(lst)-1:
        return pad
    return lst[idx+offset]

def create_window(lst, idx, pad):
    feats = []
    for offset in [-2,-1,0,1,2]:
        feats.append(apply_offset(lst, idx, offset, pad))
    return feats

for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        for i in range(len(words)):
            word_feats = create_window(words, i, "1739")
            caps_feats = create_window(caps, i, "1")
            print " ".join(word_feats+caps_feats)+" "+str(labels[i])
        words = []
        caps = []
        labels = []
        continue
    info = line.split()
    words.append(str(int(info[1])+1))
    caps.append(str(int(info[2])+1))
    labels.append(labelSet[info[3]])

