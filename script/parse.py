#!/usr/bin/python
# -*- coding:utf-8 -*-

import CRFPP
import sys
import csv
import json
import glob

train_path = "../train/"
entity_path = "../entity/"
intent_path = "../intent/"

all_entities = {}

def do_load_json(file):
    with open(file, 'r', encoding='utf-8') as jsonfile:
        js = json.load(jsonfile)
        for entity in js:
            all_entities[entity['entity']] = entity

def load_json(path, file):
    do_load_json(path + file + ".json")

def load_entity(file):
    load_json(entity_path, file)

def load_all_entity():
    files = glob.glob(entity_path + r'*.json')
    for i in files:
        do_load_json(i)

def load_intent(file):
    return load_json(intent_path, file)

def read_labels(path):
    reader = csv.reader(open(path + "labels.data"), delimiter="\t")
    result = {}
    for i in reader:
        result[i[0]] = i[1]

    return result

def parse_sentence(intent, sentence, pos = 0):
    model_path = train_path + intent['entity'] + "/"
    which_model = model_path + "model"
    arguments = "-m " + which_model + " -v 3 -n2"

    #print("parse : ", sentence, intent['entity'])

    labels = read_labels(model_path)

    result = {}

    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    tagger = CRFPP.Tagger(arguments)

    # clear internal context
    tagger.clear()

    for w in sentence:
        w.strip()
        tagger.add(w)
        #tagger.add(w.encode('utf-8'))

    # parse and change internal stated as 'parsed'
    tagger.parse()

    size = tagger.size()
    tag = 'O'
    start = 0

    for i in range(0, size):
        new_tag = tagger.y2(i)
        if tag != new_tag:
            if tag != 'O':
                result[labels[tag]] = (pos + start, i - start)
            tag = new_tag
            start = i

    if tag != 'O':
        result[labels[tag]] = (pos + start, size - start)

    #print(sentence, result)

    if "compound" in intent:
        handle_compound(intent, result, sentence, pos)
    elif "choice" in intent:
        handle_choice(intent, result, sentence, pos)
    elif "enum" in intent:
        pass

    return result

###################################################################
def handle_choice(entity, data, sentence, pos):
    if len(data) != 1:
        print("confused", result, entity, sentence)
        exit(1)

    for c in data:
        child = all_entities[c]
        if 'model' not in child or child['model']:
            data[c] = parse_sentence(child, sentence, pos)

###################################################################
def check_intent_args(entity, data):
    fields = filter(lambda x: x['mandatory'], entity['compound'])
    fields = sorted(fields, key=lambda x: x['priority'])
    for field in fields:
        if field['name'] not in data:
            print(field['question'], data)
            sys.exit(1)

###################################################################
def check_entity_args(entity, data):
    fields = filter(lambda x: x['mandatory'], entity['compound'])
    for field in fields:
        if field['name'] not in data:
            print(field['name'], "missing")
            sys.exit(1)

###################################################################
def handle_compound(entity, data, sentence, pos):
    if 'class' in entity and entity['class'] == 'intent':
        check_intent_args(entity, data)
    else:
        check_entity_args(entity, data)

    for field in entity['compound']:
        if field['name'] in data:
            child = all_entities[field['type']]
            if 'model' not in child or child['model']:
                start, length = data[field['name']]
                data[field['name']] = parse_sentence(child, sentence[start - pos :start+length - pos], start)



###################################################################
if len(sys.argv) != 3:
        sys.exit(2)

sentence = sys.argv[2]

load_all_entity()

intent_name = sys.argv[1]
load_intent(intent_name)

intent = all_entities[intent_name]

#src_data = sys.argv[2].decode('string_escape').decode('utf-8')

result = parse_sentence(intent, sentence)

result['sentence'] = sentence

print(result)
