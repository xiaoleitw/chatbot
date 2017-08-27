# chatbot

**警告**：此项目尚处于概念验证期。有很多定义方式和实现方式都尚不完善和成熟，在随后的版本中可能会随时修改。

如果你实在愿意当下就把玩一下，可按照本文档的简单说明进行尝试。

## 1. 安装

系统：`Linux` 或 `Mac`。

1. 安装`python 3.5`或更高版本。
2. 下载和安装`CRF++`[http://taku910.github.io/crfpp/]。

安装`CRF++`时，其中的`python bindings`（在目录`crf++3.5/python`下)，必需用`python3`进行安装。


## 2. 快速开始

### 2.1 生成训练数据

进入到script目录，执行:

```
python3 train-data-generator.py
```

### 2.2 训练模型

进入到script目录，执行：

```
python3 train.py
```

### 2.3 测试模型

进入到script目录，执行：

```
python3 chatbot.py
```

这是一个交互式对话环境。你可以输入任何你想说的话，模型会根据你说的话作出回应。

任何时候，可以输入`bye`，来退出交互式对话环境。

## 3. 配置

### 3.1 增加intent

在`intent`目录下创建一个`json`文件，或在已有的`json`文件里添加一个`intent`配置。

一个`intent`包含四个元素：

1. `entity`: 指定此`intent`的名字；
2. `class`: 其值必需是"intent"，以说明这是一个`intent`；
3. `compound`: 用来指定`intent`的参数列表；
4. `patterns`: 指定此`intent`的例句模式。

而每一个参数，都必须指定两个元素：

1. `name`: 此参数的名字
2. `type`: 此参数对应的`entity`的名字

除了这两个参数，其它可选参数包括：

1. `required`：指定此参数是否是必备参数；如果其值为`true`，则后续参数也必须提供。
2. `priority`：如果用户给出的句子里，多个`required`的参数缺失，`chatbot`会按照`priority`的值从小到大依次询问用户问题，以引导用户给出缺失参数；
3. `question`：当用户句子里参数缺失时，用来引导用户的问题。应至少给出一个问句，如果多于一个，每次`chatbot`会随机选择一个，从而让对话不那么枯燥。

以下是一个例子：

```
{
  "entity" : "book-ticket",
  "class"  : "intent",
  "compound" : [
    {
      "name" : "datetime",
      "type" : "datetime",
      "required" : true,
      "priority" : 2,
      "question": [
        "您要订哪天的票？",
        "请问您哪天出发？"
      ]
    },
    {
      "name" : "from",
      "type" : "general-city",
      "required" : false
    },
    {
      "name" : "to",
      "type" : "general-city",
      "required" : true,
      "priority" : 1,
      "question" : [
        "请告我您要到达的城市",
        "您要订到哪里的票？",
        "您的目的地是哪里？",
        "您要到哪个城市？"
      ]
    },
    {
      "name" : "ticket",
      "type" : "ticket",
      "required" : true,
      "priority" : 0,
      "question": [
        "请问您要订什么票？
        "]
      }
  ],
  "patterns": [
    "订张@{ticket}",
    "订@{ticket}",
    "我想订@{ticket}",
    "帮我订张@{ticket}"
  ]
}
```


### 3.2 增加entity

在`entity`目录下创建一个新的`json`文件，用来添加新的`entity`定义。或者在已有的`json`里添加。

`entity`有四种类型：

####`Compound`

定义像`intent`类似。但无需指定`class`，其参数，也无需指定`priority`和`question`。例如：

```
 {
    "entity" : "datetime",
    "compound" : [
      { "name" : "date",  "type": "any-date",     "required":true },
      { "name" : "time",  "type": "any-time",     "required":false }
    ],
    "patterns": [
      "@{date}@{time}",
      "@{date}",
      "@{time}"
    ]
  }
```
#### `Choice`

一个`Choice`类型的`entity`就像`C`语言里的`union`，即提供多种选择，但每次只可能选择其中一个。"choice"列表里，给出可供选择的其它`entity`的名字。例子如下：

```
  {
    "entity" : "date",
    "choice" : [
      "week-day",
      "relative-day",
      "regular-day"
    ]
  },
```

#### `Enum`

一个`Enum`类型的`entity`就像`C`语言里的`enum`，即给出所有可供使用的值。"choice"列表里，给出可供选择的其它`entity`的名字。例子如下：

```
{
    "entity" : "window",
    "enum" : {
      "source" : "window.csv",
      "column" : 1
    },
    model: true,
    "patterns" : [
      "@{this}窗",
      "@{this}窗户",
      "@{this}车窗"
    ]
  }
```

从例子可以看出，一个`enum`定义包含四个元素：

1. `entity`: 名字
2. `enum`: 用来指定可供使用的数据来源。其中：

    1. `source`: 用来指定存放数据的`csv`文件名。
    2. `column`: 用来指定使用`csv`文件里的哪一列作为数据源。
3. `model`: 是否生成子模型；
4. `patterns`: 从数据源提取数据后，生成最终数据的模式。其中`@{this}`指定的是从`csv`文件中读到的数据。

#### `Templates`

一个`Templates`类型的`entity`，就像`Choice`一样，提供多种选择，但每次只可能选择一个。不同的地方在于：`Choice`提供的选择是其它`entity`，而`Templates`提供的选择是`Template`。

例子如下：

```
{
    "entity" : "any-window",
    "templates": [
      { "name" : "single",         "model" : false },
      { "name" : "and-list",       "model" : true  }
    ],
    "source-type" : "window"
}
```

一个`Template`类型的包含三个元素：

1. `entity`: 名字
2. `templates`: 可供选择的`template`列表。
3. `source-type`：用来传递给模版的`entity`名字。

而每个`template`则包含两个元素：

1. `name`：引用的`template`的名字。
2. `model`：是否生成子模型。
