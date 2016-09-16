import sys, codecs

txt = sys.argv[1]
frm = sys.argv[2]

data = codecs.open(txt, 'r', 'utf-8').readlines()

ptr = 0
for line in open(frm):
    line = line.strip()
    if line.startswith('nl:'):
        print 'nl:'+data[ptr].strip().encode('utf-8')
        ptr += 1
    else:
        print line
