#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import csv
import sys
import glob
from random import choice
import random
import os
import errno

entity_path = "../entity/"
csv_path    = "../csv/"
train_path = "../train/"
intent_path = "../intent/"

all_templates = {}
csv_samples_dict = {}
entity_dict = {}

def make_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise

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
    files = glob.glob(entity_path + r'*.json') + glob.glob(intent_path + r'*.json')
    for i in files:
        do_load_json(i)


def read_chinese():
    reader = csv.reader(open("../raw-data/chinese.txt", 'r', encoding='utf-8'), delimiter=" ")
    result = []
    for i in reader:
        result += i

    return result

noise_lib = read_chinese()

def make_train_noise(size):
    num = choice(range(0, size, 1))
    return [[choice(noise_lib), 'O'] for i in range(num)]

def make_test_noise(size):
    num = choice(range(0, size, 1))
    return [[choice(noise_lib)] for i in range(num)]

################################################################################
def generate_template_data(data_list, template_name, n_samples):
    if template_name not in all_templates:
        print("template", template_name, "does not exist")
        exit(1)

    template = all_templates[template_name]
    #print(template)

################################################################################
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

label_str = ['A', 'B', 'C', 'D', 'F', 'G', 'H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']

def generate_unlabelled_sample(samples):
    all_samples = []
    for sample in samples:
        tags = sample[0]
        words = sample[1]
        result = []
        for word in words:
            result += [w for w in word]
        all_samples.append(result)


    return all_samples

def generate_labelled_sample(samples, labels):
    all_samples = []
    for sample in samples:
        tags = sample[0]
        words = sample[1]
        start = 0
        result = []
        for tag in tags:
            index = tag[0]
            for s in range(start, index):
                result += [[w, 'O'] for w in words[s]]
            result += [[w, labels[tag[1]]] for w in words[index]]
            if len(result) > 0: result[-1][1] = result[-1][1]+'E'
            start = index + 1
        for s in range(start, len(words)):
            result += [[w, 'O'] for w in words[s]]

        all_samples.append(result)

    return all_samples


def flattern_generate_sample_by_template(template_name, data_list, n_samples):
    _, samples = generate_sample_by_template(template_name, data_list, n_samples)
    return ["".join(sample[1]) for sample in samples]

def generate_labelled_classified_samples(samples, labels):
    result = []
    for sample in samples:
        label = labels[sample[0]]
        result.append([[c, label] for c in sample[1]])

    return result

def generate_classified_samples(samples):
    result = []
    for sample in samples:
        result.append([[c] for c in sample[1]])

    return result

def add_train_noise(samples):
    n = len(samples)
    samples += [make_train_noise(20) for _ in range(0, n // 10)]

    random.shuffle(samples)

    return samples[:n]

def add_train_sample_noise(samples):
    return [make_train_noise(5) + sample + make_train_noise(5) for sample in samples]

def add_test_noise(samples):
    n = len(samples)
    samples += [make_test_noise(20) for _ in range(0, n // 10)]

    random.shuffle(samples)

    return samples[:n]

def add_test_sample_noise(samples):
    return [make_test_noise(5) + sample + make_test_noise(5) for sample in samples]

################################################################################
class SampleObject:
    def generate_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_samples")

    def generate_flat_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_flat_samples")

    def generate_train_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_train_samples")

    def generate_test_samples(self, n_samples):
        raise NotImplementedError("SHOULD NOT BE HERE: generate_test_samples")

################################################################################
class EnumEntity(SampleObject):
    def __init__(self, name):
        entity = entity_dict[name]
        self.name = name
        if name not in csv_samples_dict:
            e = entity['enum']
            csv_samples_dict[name] = {'in-context':read_csv(e['source'], e['column'])}

            if 'patterns' in entity:
                samples = []
                for pattern in entity['patterns']:
                    if '@{this}' in pattern:
                        for i in csv_samples_dict[name]['in-context']:
                            s = []
                            for j in pattern:
                                if j == '@{this}':
                                    s.append(i)
                                else:
                                    s.append(j)
                            samples.append("".join(s))
                    else:
                        samples.append("".join(pattern))

                    csv_samples_dict[name]['out-context'] = samples
            else:
                csv_samples_dict[name]['out-context'] = csv_samples_dict[name]['in-context']

    def generate_samples(self, n_samples, in_context=False):
        result = [choice(csv_samples_dict[self.name]['out-context']) for i in range(n_samples)]

        if in_context:
            result += [choice(csv_samples_dict[self.name]['in-context']) for i in range(n_samples)]
            random.shuffle(result)

        return result[:n_samples]

    def generate_flat_samples(self, n_samples):
        return self.generate_samples(n_samples)

    def generate_train_samples(self, n_samples):
        samples = self.generate_samples(n_samples, in_context=True)
        result = []
        for sample in samples:
            data = [[c, 'A'] for c in sample]
            data[-1][1] = 'AE'
            result.append(data)


        return [['A', self.name]], result

    def generate_test_samples(self, n_samples):
        samples = self.generate_samples(n_samples, in_context=True)
        result = []
        for sample in samples:
            result.append([[c] for c in sample])

        return result

################################################################################
class ChoiceEntity(SampleObject):
    def __init__(self, name):
        entity = entity_dict[name]
        self.name = name
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

    def generate_train_samples(self, n_samples):
        samples = self.generate_samples(n_samples)

        labels = {}
        for c, i in zip(self.children, range(0, len(self.children))):
            labels[c] = label_str[i]

        return [[labels[c], c] for c in labels], generate_labelled_classified_samples(samples, labels)

    def generate_test_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return generate_classified_samples(samples)

################################################################################
class CompoundEntity(SampleObject):
    def __init__(self, name):
        self.name = name
        entity = entity_dict[name]
        self.templates = entity['patterns']
        self.children = {}
        for child in entity['compound']:
            self.children[child['name']] = create_entity(child['type'])

        self.result_labels = []
        self.labels = {}
        for c, i in zip(self.children, range(0, len(self.children))):
            self.labels['@{' + c + '}'] = label_str[i]
            self.result_labels.append([label_str[i], c])

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

    def generate_train_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return self.result_labels, generate_labelled_sample(samples, self.labels)

    def generate_test_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return generate_unlabelled_sample(samples)

################################################################################
class TemplateEntity(SampleObject):
    def __init__(self, name):
        entity = entity_dict[name]
        self.name = name
        self.child = create_entity(entity['source-type'])
        self.templates = entity['templates']

        self.labels = {}
        for c, i in zip(self.templates, range(0, len(self.templates))):
            self.labels[c['name']] = label_str[i]

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

    def __generated_label_table(self):
        return [[self.labels[c], c] for c in self.labels]

    def generate_train_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return self.__generated_label_table(), generate_labelled_classified_samples(samples, self.labels)

    def generate_test_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return generate_classified_samples(samples)

######################################################################################################
class Template(SampleObject):
    def __init__(self, name, parent):
        self.name = parent + '/' + name
        self.raw_name = name

        template = all_templates[name]

        self.result_labels = []
        self.labels = {}
        for p, i in zip(template['parameters'], range(0, len(template['parameters']))):
            self.labels['@{' + p + '}'] = label_str[i]
            self.result_labels.append([label_str[i], p])

        self.data_src = create_entity(entity_dict[parent]['source-type'])

    def generate_samples(self, n_samples):
        data_list = self.data_src.generate_flat_samples(n_samples * 2)
        _, samples = generate_sample_by_template(self.raw_name, data_list, n_samples)
        return samples

    def generate_flat_samples(self, n_samples):
        data_list = self.data_src.generate_flat_samples(n_samples * 2)
        return flattern_generate_sample_by_template(self.raw_name, data_list, n_samples)

    def generate_train_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return self.result_labels, generate_labelled_sample(samples, self.labels)

    def generate_test_samples(self, n_samples):
        samples = self.generate_samples(n_samples)
        return generate_unlabelled_sample(samples)


################################################################################
load_all_entity()

def __create_entity(name):
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

def create_entity(name):
    if name in entity_dict:
        return __create_entity(name)

    print("no entity: ", name)
    exit(1)

def do_write_csv_data(entity_name, samples, ty):
    path = train_path + entity_name
    make_path(path)
    with open(path + "/" + ty + '.data', 'w', encoding='utf-8') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',lineterminator='\n')
        for i in samples:
            my_writer.writerow(i)

def write_samples(entity_name, samples, ty):
    print("writing: " + entity_name)
    path = train_path + entity_name
    make_path(path)
    with open(path + "/" + ty + '.data', 'w', encoding='utf-8') as csvfile:
        my_writer = csv.writer(csvfile, delimiter='\t',lineterminator='\n')
        for sample in samples:
            for i in sample:
                my_writer.writerow(i)
            my_writer.writerow([""])

def generate_train_samples(obj, n_samples, noise=False, sample_noise=False):
    labels, samples = obj.generate_train_samples(n_samples)

    sentence = [["".join([i[0] for i in s])] for s in samples]

    if sample_noise: samples = add_train_sample_noise(samples)
    if noise: samples = add_train_noise(samples)

    do_write_csv_data(obj.name, labels, 'labels')
    do_write_csv_data(obj.name, sentence, 'sentence')
    write_samples(obj.name, samples, 'train')

def generate_test_samples(obj, n_samples, noise=False, sample_noise=False):
    samples = obj.generate_test_samples(n_samples)
    if sample_noise: samples = add_test_sample_noise(samples)
    if noise: samples = add_test_noise(samples)
    write_samples(obj.name, samples, 'test')

def do_generate_artifacts(obj, n_train, n_test, noise=False, sample_noise=False):
    generate_train_samples(obj, n_train, noise, sample_noise)
    generate_test_samples(obj, n_test, noise, sample_noise)

def make_entity_training_artifacts(entity_name, n_train, n_test, noise=False, sample_noise=True):
    entity = create_entity(entity_name)
    do_generate_artifacts(entity, n_train, n_test, noise, sample_noise)

def make_template_training_artifacts(template_name, parent_name, n_train=1000, n_test=10, noise=False, sample_noise=True):
    template = Template(template_name, parent_name)
    do_generate_artifacts(template, n_train, n_test, noise, sample_noise)

################################################################################
#entity_name = "entry"
#entity = create_entity(entity_name)
#samples = entity.generate_test_samples(10000)
#print(["".join([i[0] for i in s]) for s in samples])

#print(["".join(s) for s in samples])

make_entity_training_artifacts('entry', 30000, 1000, noise=True)

make_entity_training_artifacts('welcome', 100, 10)
make_entity_training_artifacts('open-door', 10000, 100)
make_entity_training_artifacts('close-door', 10000, 100)
make_entity_training_artifacts('open-window', 10000, 100)
make_entity_training_artifacts('close-window', 10000, 100)
make_entity_training_artifacts('adjust-window-up',   10000, 100)
make_entity_training_artifacts('adjust-window-down', 10000, 100)
make_entity_training_artifacts('scale-window-up',      10000, 100)
make_entity_training_artifacts('scale-window-down',      10000, 100)
make_entity_training_artifacts('play-music', 10000, 100, sample_noise=False)
make_entity_training_artifacts('book-ticket', 20000, 100)

make_template_training_artifacts('centered-range', 'any-date', 10000)
make_template_training_artifacts('range', 'any-date', 10000)
make_template_training_artifacts('or-list', 'any-date', 10000)
make_template_training_artifacts('and-list', 'any-date', 10000)

make_template_training_artifacts('centered-range', 'any-time', 10000)
make_template_training_artifacts('range', 'any-time', 10000)

make_template_training_artifacts('centered-range', 'rough-scale', 1000)

make_template_training_artifacts('and-list', 'multi-window', 1000)

make_entity_training_artifacts('any-time', 10000, 100, noise=True, sample_noise=False)
make_entity_training_artifacts('any-date', 10000, 100, noise=True, sample_noise=False)
make_entity_training_artifacts('relative-day', 1000, 100, noise=True)

make_entity_training_artifacts('time', 20000, 100)
make_entity_training_artifacts('date', 20000, 100, noise=True, sample_noise=False)
make_entity_training_artifacts('regular-day', 20000, 100, noise=True)
make_entity_training_artifacts('regular-month', 20000, 100, noise=True)
make_entity_training_artifacts('ticket', 1000, 100, noise=True)
make_entity_training_artifacts('general-city', 10000, 100,  noise=True)
make_entity_training_artifacts('province-city', 10000, 100, noise=True)
make_entity_training_artifacts('specific-window', 1000, 100)
make_entity_training_artifacts('any-window', 1000, 100, noise=True)
make_entity_training_artifacts('singer', 1000, 100, sample_noise=False)

make_entity_training_artifacts('rough-scale', 1000, 100)
make_entity_training_artifacts('scale', 1000, 100, noise=True)

#entity = create_entity('book-ticket')
#print(entity.generate_train_samples(1))
