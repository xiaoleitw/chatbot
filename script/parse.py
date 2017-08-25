#!/usr/bin/python
# -*- coding:utf-8 -*-

import CRFPP
import sys
import csv
import json
import glob
import pprint
from random import choice

from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

history = InMemoryHistory()

train_path = "../train/"
entity_path = "../entity/"
intent_path = "../intent/"

all_entities  = {}
all_templates = {}

################################################################################
def do_load_json(file):
    with open(file, 'r', encoding='utf-8') as jsonfile:
        js = json.load(jsonfile)
        for entity in js:
            if 'entity' in entity:
                all_entities[entity['entity']] = entity
            elif 'template' in entity:
                all_templates[entity['template']] = entity

################################################################################
def load_json(path, file):
    do_load_json(path + file + ".json")

################################################################################
def load_entity(file):
    load_json(entity_path, file)

################################################################################
def load_all_entity():
    files = glob.glob(entity_path + r'*.json')
    for i in files:
        do_load_json(i)

def load_all_intent():
    files = glob.glob(intent_path + r'*.json')
    for i in files:
        do_load_json(i)

################################################################################
def load_intent(file):
    return load_json(intent_path, file)

################################################################################
def read_labels(path):
    reader = csv.reader(open(path + "labels.data"), delimiter="\t")
    result = {}
    for i in reader:
        result[i[0]] = i[1]

    return result

def get_param(sentence, start, pos, end):
    s = pos + start
    l = end - start
    return ((s, l), sentence[start:end])

################################################################################
class Parser:
    def __init__(self, name):
        model_path = train_path + name + "/"
        which_model = model_path + "model"
        arguments = "-m " + which_model + " -v 3 -n2"
        self.labels = read_labels(model_path)
        self.tagger = CRFPP.Tagger(arguments)
        self.name = name

    def get_name(self):
        return self.name

    ############################################################################
    def parse_content(self, sentence, pos, container):
        size = self.tagger.size()
        tag = 'O'
        start = 0

        for i in range(0, size):
            new_tag = self.tagger.y2(i)
            if len(new_tag) > 0 and new_tag[-1] == 'E':
                new_tag = new_tag[:-1]

            if tag != new_tag:
                if tag != 'O':
                    container(tag, get_param(sentence, start, pos, i))

                tag = new_tag
                start = i

        if tag != 'O':
            container(tag, get_param(sentence, start, pos, size))
            #container(tag, (pos + start, size - start))
            #@result[self.labels[tag]] = (pos + start, size - start)

    ############################################################################
    def parse_dict(self, sentence, pos):
        result = {}

        def assign(tag, pos): result[self.labels[tag]] = pos

        self.parse_content(sentence, pos, assign)

        return result

    ############################################################################
    def parse_sentence(self, sentence, pos):
        self.tagger.clear()

        for w in sentence:
            w.strip()
            self.tagger.add(w)
            #tagger.add(w.encode('utf-8'))

        # parse and change internal stated as 'parsed'
        self.tagger.parse()

        return self.do_parse(sentence, pos)


    def do_parse(self, sentence, pos):
        raise NotImplementedError("SHOULD NOT BE HERE: do_parse")


    def verify(self, intent):
        return True, intent
        #raise NotImplementedError("SHOULD NOT BE HERE: verify")


################################################################################
class IntentParser(Parser):
    def __init__(self, intent):
        super().__init__(intent['entity'])
        self.entity = intent

    def do_parse(self, sentence, pos):
        return self.deep_parse(self.parse_dict(sentence, pos), sentence, pos)

    def deep_parse(self, result, sentent, pos):
        raise NotImplementedError("SHOULD NOT BE HERE: deep_parse")


class CompoundIntentParser(IntentParser):
    def __init__(self, intent):
        super().__init__(intent)
        self.fields = filter(lambda x: 'required' in x and x['required'], self.entity['compound'])
        if 'class' in self.entity and self.entity['class'] == 'intent':
            self.fields = sorted(self.fields, key=lambda x: x['priority'])

        self.children = {}

        for field in self.entity['compound']:
            child = all_entities[field['type']]
            if 'model' not in child or child['model']:
                parser = get_parser(child['entity'])
                if not parser:
                    print("null parser: ", child['entity'])
                    exit(1)
                else:
                    self.children[field['name']]  = parser

    def verify(self, intent):
        if self.entity['class'] == 'intent':
            for f in self.fields:
                if f['name'] not in intent:
                    return False, {'slot':{'type': f['type'], 'name':f['name']}, 'reply': choice(f['question'])}

        return True, intent

    def deep_parse(self, data, sentence, pos):
        for child in self.children:
            if child in data:
                start, l = data[child][0]
                parser = self.children[child]
                if parser == None:
                    print("null parser: ", child)
                    exit(1)

                s = start - pos
                data[child] = parser.parse_sentence(sentence[s : s + l], start)

        return data

###############################################################################
class ChoiceIntentParser(IntentParser):
    def __init__(self, intent):
        super().__init__(intent)

        self.children = {}
        for field in self.entity['choice']:
            child = all_entities[field]
            if 'model' not in child or child['model']:
                sub_parser = get_parser(field)
                if sub_parser:
                    self.children[field] = get_parser(field)
                else:
                    print(field, "create failed")
                    exit(1)

    def deep_parse(self, data, sentence, pos):
        if len(data) != 1:
            print("ChoiceIntentParser confused: ", data, sentence)
            exit(1)

        for c in data:
            if c in self.children:
                data[c] = self.children[c].parse_sentence(sentence, pos)

        return data

###############################################################################
class EnumIntentParser(IntentParser):
    def __init__(self, intent):
        super().__init__(intent)

    def deep_parse(self, data, sentence, pos):
        return data

###############################################################################
class TemplateIntentParser(IntentParser):
    def __init__(self, intent):
        super().__init__(intent)

        self.parser = get_parser(intent['source-type'])

        self.children = {}
        for template in intent['templates']:
            if 'model' in template and template['model']:
                self.children[template['name']] = create_template_parser(template['name'], intent['entity'], self.parser)

    def deep_parse(self, data, sentence, pos):
        if len(data) != 1:
            print("TemplateIntentParser confused: ", data, sentence)
            exit(1)

        for c in data:
            if c in self.children:
               data[c] = self.children[c].parse_sentence(sentence, pos)
            else:
               s, l = data[c][0]
               data[c] = self.parser.parse_sentence(sentence[:l], pos)

        return data

###############################################################################
class TemplateParser(Parser):
    def __init__(self, template, parent, parser):
        super().__init__(parent + "/" + template['template'])
        self.result_type = template['result-type']
        self.parser = parser


    def __parse_list(self, sentence, pos):
        result = []

        def assign(tag, pos): result.append(pos)

        self.parse_content(sentence, pos, assign)

        return result

    def __parse_value(self, sentence, pos):
        value = []

        def assign(tag, pos): value.append(pos)

        self.parse_content(sentence, pos, assign)

        return value[0]

    def handle_value(self, sentence, pos):
        value = self.__parse_value(sentence, pos)
        if not self.parser:
            return value

        s = pos + value[0]

        return self.parser.parse_sentence(sentence[s:s+value[1]], s)

    def __parse(self, sentence, pos, value):
        s = value[0]
        return self.parser.parse_sentence(sentence[s[0] - pos : s[0] - pos + s[1]], s[0])

    def handle_list(self, sentence, pos):
        values = self.__parse_list(sentence, pos)
        if not self.parser:
            return values

        return [self.__parse(sentence, pos, i) for i in values]

    def handle_dict(self, sentence, pos):
        data = self.parse_dict(sentence, pos)
        if not self.parser:
            return data

        for c in data:
            v = data[c][0]
            data[c] = self.parser.parse_sentence(sentence[v[0] - pos : v[0] - pos + v[1]], v[0])

        return data

    def do_parse(self, sentence, pos):
        if 'value' == self.result_type:
            return self.handle_value(sentence, pos)
        elif 'list' == self.result_type:
            return self.handle_list(sentence, pos)
        elif 'dict' == self.result_type:
            return self.handle_dict(sentence, pos)

###################################################################
def handle_choice(entity, data, sentence, pos):
    if len(data) != 1:
        print("handle_choice confused: ", data, sentence)
        exit(1)

    for c in data:
        child = all_entities[c]
        if 'model' not in child or child['model']:
            data[c] = parse_sentence(child, sentence, pos)

###################################################################
def check_intent_args(entity, data):
    fields = filter(lambda x: x['required'], entity['compound'])
    fields = sorted(fields, key=lambda x: x['priority'])
    for field in fields:
        if field['name'] not in data:
            print(field['question'], data)
            sys.exit(1)

###################################################################
def check_entity_args(entity, data):
    fields = filter(lambda x: 'required' in x and x['required'], entity['compound'])
    for field in fields:
        if field['name'] not in data:
            print(field['name'], "missing:", data)
            sys.exit(1)

###################################################################
all_entity_parser = {}
all_template_parser = {}

def create_parser(name):
    entity = all_entities[name]

    if 'compound' in entity:
        return CompoundIntentParser(entity)
    elif 'choice' in entity:
        return ChoiceIntentParser(entity)
    elif 'enum' in entity:
        return EnumIntentParser(entity)
    elif 'templates' in entity:
        return TemplateIntentParser(entity)

    return None

def get_entity_parser(name):
    if name in all_entity_parser:
        return all_entity_parser[name]

    parser = create_parser(name)
    if parser: all_entity_parser[name] = parser

    return parser

def create_template_parser(name, parent, parser):
    template = all_templates[name]
    if not template:
        print("error")
        exit(1)

    return TemplateParser(template, parent, parser)

def get_parser(name):
    if name not in all_entities:
        print("error")
        exit(1)

    return get_entity_parser(name)

################################################################################
def do_check_intent(intent, name):
    s, r = get_parser(name).verify(intent)
    if s:
        pp.pprint({name : intent})
    else:
        #{'type': f['type'], 'reply': f['question']}
        pp.pprint(r['reply'])
        context['intent'] = intent
        context['intent-name'] = name
        context['slot'] = r['slot']
        context['parser'].append(get_parser(r['slot']['type']))
        #pp.pprint({name:intent})

def check_intent(intent):
    if len(intent) != 1:
        print("this is not a intent")
        exit(1)

    for c in intent:
        do_check_intent(intent[c], c)
            #context['parser'].append()

def fill_slot(result):
    context['intent'][context['slot']['name']] = result

    name   = context['intent-name']
    intent = context['intent']

    context['parser'] = context['parser'][:-1]
    context['slot'] = ""
    context['intent'] = None
    context['intent-name'] = ""

    do_check_intent(intent, name)

################################################################################

load_all_entity()
load_all_intent()
context = {'parser':[], 'intent':None, 'name':'', 'type':""}

pp = pprint.PrettyPrinter(depth=10)

#print(all_templates)

if __name__ == "__main__":
    context['parser'].append(get_parser('entry'))

    while True:
        sentence = prompt('> ', history=history, auto_suggest=AutoSuggestFromHistory())
        result = context['parser'][-1].parse_sentence(sentence, 0)
        if context['intent'] == None:
            check_intent(result)
        else:
            fill_slot(result)

        #pp.pprint(result)
