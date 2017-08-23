#!/usr/bin/python
# -*- coding:utf-8 -*-

import CRFPP
import sys
import csv

csv_path = "./"

def read_csv(file):
    reader = csv.reader(open(csv_path + file, 'r', encoding='utf-8'), delimiter="\t")
    result = []
    for i in reader:
        result.append(i)

    return result

def read_labels():
    return read_csv("labels.data")

if len(sys.argv) != 2:
        sys.exit(2)

src_data = sys.argv[1]

# -v 3: access deep information like alpha,beta,prob
# -nN: enable nbest output. N should be >= 2
tagger = CRFPP.Tagger("-m ./model -v 3 -n2")

# clear internal context
tagger.clear()

for w in src_data:
   w.strip()
   tagger.add(w)

# parse and change internal stated as 'parsed'
tagger.parse()

print(read_labels())
print("#" , tagger.prob())

size = tagger.size()
xsize = tagger.xsize()
for i in range(0, size):
   print(tagger.x(i, 0) , "\t", tagger.y2(i)+"／"+str(tagger.prob(i)))

