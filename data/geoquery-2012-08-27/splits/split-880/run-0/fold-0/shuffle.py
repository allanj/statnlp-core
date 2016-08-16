import sys, time, random
random.seed(time.time())
lines = open(sys.argv[1]).readlines()
#random.shuffle(lines)
lines = lines[:int(sys.argv[2])]
lines = sorted(lines, key=int)
for l in lines:
    print l.strip()
