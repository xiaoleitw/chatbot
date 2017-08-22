#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import csv
import sys
import glob
from random import choice
import random

entity_path = "../entity/"
csv_path    = "../csv/"

all_templates = {}
csv_samples_dict = {}
entity_dict = {}

############################################################################
def split_pattern(pattern):
    pos = pattern.find('@{')
    if pos < 0:
        return [pattern]

    end = pattern.find('}')

    result = []
    if pos > 0:
        result += [pattern[:pos]]

    result += [pattern[pos:end+1]]

    if end < len(pattern) - 1:
        result += split_pattern(pattern[end+1:])

    return  result

def process_pattern_part(part):
    for i in part:
        part[i] = split_pattern(part[i])

    return part

def process_pattern(pattern):
    return [process_pattern_part(p) for p in pattern]

def process_patterns(patterns):
    return [ process_pattern(p) for p in patterns ]

############################################################################
def load_json(path, file):
    with open(path + file + ".json", 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)


def read_csv(file, column):
    reader = csv.reader(open(csv_path + file, 'r', encoding='utf-8'), delimiter="\t")
    result = []
    for i in reader:
        result.append(i[column-1])

    return result

def do_load_json(file):
    with open(file, 'r', encoding='utf-8') as jsonfile:
        js = json.load(jsonfile)
        for obj in js:
            if 'entity' in obj:
                entity_dict[obj['entity']] = obj
                if 'patterns' in obj:
                    obj['patterns'] = [split_pattern(i) for i in obj['patterns']]
            elif 'template' in obj:
                obj['patterns'] = process_patterns(obj['patterns'])
                all_templates[obj['template']] = obj

def load_json(path, file):
    do_load_json(path + file + ".json")

def load_entity(file):
    load_json(entity_path, file)

def load_all_entity():
    files = glob.glob(entity_path + r'*.json')
    for i in files:
        do_load_json(i)


######################################################################################################
def generate_template_data(data_list, template_name, n_samples):
    if template_name not in all_templates:
        print("template", template_name, "does not exist")
        exit(1)

    template = all_templates[template_name]
    #print(template)

######################################################################################################
#print(all_templates['or-list'])
class SingleSampleGenerator:
    def __init__(self, pattern, data_list, param_list):
        self.pattern = pattern
        self.data_list = data_list
        self.param_list = param_list
        self.index = 0
        self.sample = []
        self.label_list = []

    def generate_one_item(self, tag):
        result = []
        if tag in self.param_list:
            self.label_list.append((self.index, tag))
            result = choice(self.data_list)
        else:
            result = tag

        self.index += 1

        return result

    def __once(self, part):
        self.sample += [self.generate_one_item(i) for i in part['once']]

    def __many(self, part):
        times = choice(range(0,3,1))
        for j in range(times):
            self.sample += [self.generate_one_item(i) for i in part['many']]


    def generate(self):
        for part in self.pattern:
            if 'once' in part: self.__once(part)
            if 'many' in part: self.__many(part)

        return self.label_list, self.sample


def generate_sample_by_template(template_name, data_list, n_samples):
    template = all_templates[template_name]

    param_list = {}
    for i in template['parameters']:
        param_list['@{' + i + "}"] = 0

    samples = []
    for i in range(n_samples):
        for pattern in template['patterns']:
            samples.append(SingleSampleGenerator(pattern, data_list, param_list).generate())

    random.shuffle(samples)

    return template_name, samples[:n_samples]

label_str = ['A', 'B', 'C', 'D', 'E']

def generate_unlabelled_sample(samples):
    result = []
    for sample in samples:
        tags = sample[0]
        words = sample[1]
        for word in words:
            result += [w for w in word]
        result += [['。']]

    return result

def generate_labelled_sample(samples, labels):
    result = []
    for sample in samples:
        tags = sample[0]
        words = sample[1]
        start = 0
        for tag in tags:
            index = tag[0]
            for s in range(start, index):
                result += [(w, 'O') for w in words[s]]
            result += [(w, labels[tag[1]]) for w in words[index]]
            start = index + 1
        for s in range(start, len(words)):
            result += [(w, 'O') for w in words[s]]
        result += [['。', 'O']]

    return result

def label_template_samples(samples):
    template = all_templates[samples[0]]

    labels = {}
    for p, i in zip(template['parameters'], range(0, len(template['parameters']))):
        labels['@{' + p + '}'] = label_str[i]

    return generate_labelled_sample(samples[1], labels)

def flattern_generate_sample_by_template(template_name, data_list, n_samples):
    _, samples = generate_sample_by_template(template_name, data_list, n_samples)
    return ["".join(sample[1]) for sample in samples]

def add_suffix_to_sample(sample):
    return sample + [['。', 'O']]

def generate_labelled_classified_samples(samples, labels):
    result = []
    for sample in samples:
        label = labels[sample[0]]
        result += add_suffix_to_sample([[c, label] for c in sample[1]])

    return result

def generate_classified_samples(samples):
    result = []
    for sample in samples:
        result += [[c] for c in sample[1]]
        result += [['。']]

    return result

######################################################################################################
class Entity:
    def generate_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_samples")
        #print("SHOULD NOT BE HERE: generate_samples")


    def generate_flat_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_flat_samples")


    def generate_train_samples(self, n_samples, add_noise=False):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_train_samples")

    def generate_test_samples(self, n_samples, add_noise=False):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_test_samples")

######################################################################################################
class EnumEntity(Entity):
    def __init__(self, name):
        entity = entity_dict[name]
        self.name = name
        if name not in csv_samples_dict:
            e = entity['enum']
            csv_samples_dict[name] = read_csv(e['source'], e['column'])

    def generate_samples(self, n_samples):
        return [choice(csv_samples_dict[self.name]) for i in range(n_samples)]

    def generate_flat_samples(self, n_samples):
        return self.generate_samples(n_samples)

    def generate_train_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)
        result = []
        for sample in samples:
            result += add_suffix_to_sample([[c, 'A'] for c in sample])
        return result

    def generate_test_samples(self, n_samples, add_noise=False):
        amples = self.generate_samples(n_samples)
        result = []
        for sample in samples:
            result += add_suffix_to_sample([[c] for c in sample])

        return result

######################################################################################################
class ChoiceEntity(Entity):
    def __init__(self, name):
        entity = entity_dict[name]
        self.children = {}
        for child in entity['choice']:
            self.children[child] = create_entity(child)

    def generate_samples(self, n_samples):
        n = n_samples // len(self.children)
        n += 1

        samples = []
        for c in self.children:
            ss = self.children[c].generate_flat_samples(n)
            samples += [(c, s) for s in ss]

        random.shuffle(samples)

        return samples[:n_samples]

    def generate_flat_samples(self, n_samples):
        return [s[1] for s in self.generate_samples(n_samples)]

    def generate_train_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)

        labels = {}
        for c, i in zip(self.children, range(0, len(self.children))):
            labels[c] = label_str[i]

        return generate_labelled_classified_samples(samples, labels)

    def generate_test_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)
        return generate_classified_samples(samples)

######################################################################################################
class CompoundEntity(Entity):
    def __init__(self, name):
        entity = entity_dict[name]
        self.templates = entity['patterns']
        self.children = {}
        for child in entity['compound']:
            self.children[child['name']] = create_entity(child['type'])

    def __make_sample(self, template, sub_samples, n):
        sample = []
        labels = []
        for part, index in zip(template, range(0, len(template))):
            if part in sub_samples:
                sample.append(sub_samples[part][n])
                labels.append((index, part))
            else:
                sample.append(part)

        #print(labels, sample)
        return labels, sample

    def generate_samples(self, n_samples):
        sub_samples = {}
        for c in self.children:
            sub_samples['@{' + c + '}'] = self.children[c].generate_flat_samples(n_samples)

        samples = []
        #print(n_samples)
        for i in range(0, n_samples):
            for template in self.templates:
                samples.append(self.__make_sample(template, sub_samples, i))

        random.shuffle(samples)

        return samples[:n_samples]

    def generate_flat_samples(self, n_samples):
        return ["".join(i[1]) for i in self.generate_samples(n_samples)]

    def generate_train_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)

        labels = {}
        for c, i in zip(self.children, range(0, len(self.children))):
            labels['@{' + c + '}'] = label_str[i]

        return generate_labelled_sample(samples, labels)

    def generate_test_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)
        return generate_unlabelled_sample(samples)

######################################################################################################
class TemplateEntity(Entity):
    def __init__(self, name):
        entity = entity_dict[name]
        self.child = create_entity(entity['source-type'])
        self.templates = entity['templates']

    def generate_samples(self, n_samples):
        src_list = self.child.generate_flat_samples(n_samples)

        samples = []
        n = 1 + (n_samples // len(self.templates))
        for template in self.templates:
            tmpl = template['name']
            samples += [(tmpl, i) for i in flattern_generate_sample_by_template(tmpl, src_list, n)]

        random.shuffle(samples)

        return samples[:n_samples]

    def generate_flat_samples(self, n_samples):
        return [ s[1] for s in self.generate_samples(n_samples)]

    def generate_train_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)

        labels = {}
        for c, i in zip(self.templates, range(0, len(self.templates))):
            labels[c['name']] = label_str[i]

        return generate_labelled_classified_samples(samples, labels)

    def generate_test_samples(self, n_samples, add_noise=False):
        samples = self.generate_samples(n_samples)
        return generate_classified_samples(samples)

######################################################################################################
def create_entity(name):
    if name not in entity_dict:
        print("ERROR: ", name, "does not exit")
        return None

    entity = entity_dict[name]
    if 'enum' in entity:
        return EnumEntity(name)
    elif 'choice' in entity:
        return ChoiceEntity(name)
    elif 'compound' in entity:
        return CompoundEntity(name)
    elif 'templates' in entity:
        return TemplateEntity(name)

    return None

load_all_entity()
entity = create_entity('any-date')
print(entity.generate_test_samples(20))
