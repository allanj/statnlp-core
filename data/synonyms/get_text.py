# -*- coding: utf-8 -*-
import re, sys

data = "".join(open(sys.argv[1]).readlines())
lang = sys.argv[2]
# <nl lang="id">...</nl>
data = data.decode('utf-8')
data = data.replace('\n','')
for entry in re.findall('<nl lang="'+lang+'">(.*?)'+'</nl>', data):
    out = entry.strip().replace(u'\u200b','').replace(u'\ufeff','')
    if lang == "fa":
        if out.endswith('.'):
            out = out[:-1]+" ."
    out = ' '.join(out.split())
    print out.encode('utf-8')
