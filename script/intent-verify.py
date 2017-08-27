#!/usr/bin/python
# -*- coding:utf-8 -*-

import CRFPP
import csv
import sys

labels = {}
failed_sentence = 0

def read_verify_data(file):
    reader = csv.reader(open("../train/entry/" + file + ".csv", 'r', encoding='utf-8'), delimiter="\t")
    return [i[0] for i in reader]


def read_labels():
    reader = csv.reader(open("../train/entry/" + "labels.data"), delimiter="\t")
    result = {}
    for i in reader:
        result[i[0]] = i[1]

    return result

def get_label(name):
    for i in labels:
        if labels[i] == name:
            return i

    return 'O'

def parse_sentence(sentence, label):
    global failed_sentence

    tagger.clear()

    for w in sentence:
        w.strip()
        tagger.add(w)

    tagger.parse()

    tag = 'O'
    start = 0
    tags = {}
    for i in range(0, tagger.size()):
        new_tag = tagger.y2(i)
        if tag != new_tag:
            if tag != 'O':
                if tag in tags:
                    tags[tag] += (i - start)
                else:
                    tags[tag] = (i - start)

            start = i
            tag = new_tag

    if tag != 'O':
        if tag in tags:
            tags[tag] += (tagger.size() - start)
        else:
            tags[tag] = (tagger.size() - start)

    if len(tags) != 1:
        failed_sentence += 1
        print("wrong:", tags, sentence, "total_fail:", failed_sentence)
    else:
        for c in tags:
            if c != label:
                failed_sentence += 1
                print("failed:", labels[c], "!=", label, sentence, "total_fail:", failed_sentence)
            else:
                if tagger.prob() < 0.5:
                    print(tagger.prob(), 'c', labels[c], sentence)


################################################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(1)

    labels = read_labels()

    file = sys.argv[1]
    label = get_label(file)
    if label == 'O':
        print("label for", file, "not found", )
        exit(1)


    sentences = read_verify_data(file)

    tagger = CRFPP.Tagger("-m ../train/entry/model -v 3 -n2")


    for i in sentences:
        parse_sentence(i, label)
