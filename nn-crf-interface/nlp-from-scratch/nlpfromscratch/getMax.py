import sys
numLabel = 9
weight = {}
for line in open(sys.argv[1]):
    line = line.strip()
    info = line.split(" ")
    key = "_".join(info[:10])
    weight[key] = [float(info[i]) for i in range(10,10+numLabel)]
for line in open(sys.argv[2]):
    line = line.strip()
    info = line.split(" ")
    key = "_".join(info[:10])
    max_i = 0
    scores = weight[key]
    max_val = scores[0]
    for i in range(numLabel):
        if scores[i] > max_val:
            max_val = scores[i]
            max_i = i
    print max_i+1
