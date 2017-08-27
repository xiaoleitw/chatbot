# chatbot

## 安装

系统：Linux 或 Mac。

1. 安装python3.5或更高版本。
2. 下载和安装CRF++[http://taku910.github.io/crfpp/]。

安装CRF++时，其中的python bindings（在目录crf++3.5/python下)，必需用python3进行安装。


# 快速开始

1. 生成训练数据

进入到script目录，执行:

```
python3 train-data-generator.py
```

2. 训练模型

进入到script目录，执行：

```
python3 train.py
```

3. 测试模型

进入到script目录，执行：

```
python3 chatbot.py
```

这是一个交互式对话环境。你可以输入任何你想说的话，模型会根据你说的话作出回应。

任何时候，可以输入`bye`，来退出交互式对话环境。

## 配置

### 增加intent
### 增加entity
