import sys

words = []
caps = []
labels = []

idx2label = {}
labelFile = "/Users/nlp/Documents/workspace/semantic/statnlp-core/nn-crf-interface/nlp-from-scratch/senna-torch/senna/hash/ner.lst"
for line in open(labelFile):
    line = line.strip()
    if len(idx2label) < 17:
        #if line[0] == 'S':
        #    line = 'B'+line[1:]
        #if line[0] == 'E':
        #    line = 'I'+line[1:]
        idx2label[len(idx2label)+1] = line

pred = []
for line in open(sys.argv[2]):
    pred.append(idx2label[int(line)])

pred_i = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith('-DOCSTART-'):
        print line, "O"
        continue
    if not line:
        print
        continue
    print line, pred[pred_i]
    pred_i += 1

