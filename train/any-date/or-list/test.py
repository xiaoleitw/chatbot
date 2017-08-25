#!/usr/bin/python
# -*- coding:utf-8 -*-

import CRFPP
import sys

if len(sys.argv) != 2:
        sys.exit(2)

src_data = sys.argv[1].decode('string_escape').decode('utf-8')

try:
    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    tagger = CRFPP.Tagger("-m ./model -v 3 -n2")

    # clear internal context
    tagger.clear()

    for w in src_data:
        w.strip()
        tagger.add(w.encode('utf-8'))

    # parse and change internal stated as 'parsed'
    tagger.parse()

    print "#" , tagger.prob()

    size = tagger.size()
    xsize = tagger.xsize()
    for i in range(0, size):
       for j in range(0, xsize):
          print tagger.x(i, j) , "\t",
       print tagger.y2(i)+"／"+str(tagger.prob(i))

except RuntimeError, e:
        print "RuntimeError: ", e
