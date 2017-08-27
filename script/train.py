#!/usr/bin/python
# -*- coding:utf-8 -*-
import glob
import inspect
import os
import sys

train_path = "../train"
train_cmd = "crf_learn -p2 -f 3 -c 4.0"

if len(sys.argv) > 1:
   train_path = [train_path + "/" + s for s in sys.argv[1:]]
else:
   train_path = [train_path]

files = []
for p in train_path:
    files += glob.glob(p + '/**/train.data', recursive=True)

paths = [os.path.dirname(os.path.abspath(i)) for i in files]

for path in paths:
    print("training", path)

    model = path + "/model"
    if os.path.exists(model):
        os.system("rm " + model)

    template = path + "/template"
    if not os.path.exists(template):
        template = "./template"

    train_data = path + "/train.data"

    os.system(" ".join([train_cmd, template, train_data, model]))
