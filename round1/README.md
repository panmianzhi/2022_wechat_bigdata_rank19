# 环境依赖

操作系统：

```shell
cat /proc/version
# Linux version 4.15.0-45-generic (buildd@lcy01-amd64-027) (gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.10)) #48~16.04.1-Ubuntu SMP Tue Jan 29 18:03:48 UTC 2019
```

python 3.7.11

CUDA Version: 10.1

第三方依赖库：见 requirements.txt

使用的开源预训练模型有三个，分别为[roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main), [mecbert](https://huggingface.co/hfl/chinese-macbert-base/tree/main), [nezha](https://huggingface.co/peterchou/nezha-chinese-base/tree/main)，下载模型相关文件后保存到`src/data/`的相应目录下即可。

# 代码结构

```shell
|-- config.py # 参数配置
|-- data # 存放开源预训练模型以及数据处理模块
|   |-- category_id_map.py
|   |-- macbert-base # 存放中文预训练模型macbert
|   |   |-- README.md
|   |-- roberta-wwm-ext # 存放中文预训练模型roberta
|   |   |-- README.md
|   |-- data_helper.py # 微调Dataloader接口
|   |-- lxrt_dataset.py # 预训练Dataloader接口
|   |-- nezha-chinese-base # 存放中文预训练模型nezha
|       |-- README.md
|-- kfold_finetune.py # k折微调
|-- kfold_inference.py # 集成推断，生成result文件
|-- models # 模型架构代码
|   |-- file_utils.py # bert预训练模型相关 
|   |-- finetune_model.py # 微调模型
|   |-- lxrt.py # 双流模型，ignore it
|   |-- optimization.py # 预训练优化器
|   |-- uniter.py # 预训练模型
|-- pretrain.py # 预训练代码
|-- tricks # 微调阶段tricks
|   |-- EMA.py
|   |-- FGM.py
|-- util.py # 工具代码
```

# 模型介绍

模型采用了`bert-base`架构，文本（title + asr + ocr 然后直接截断）、视频帧分别embedding后拼接，直接输入模型。模型总体采用“预训练-微调”的学习范式，预训练周期为40，文本长度截断为256。

预训练采用masked language model，image-text match，masked region feature regression三个任务，各个任务有各自的权重占比。

微调阶段：采用了预训练模型 + 双层MLP架构，使用了EMA+FGM两个tricks，进行5折交叉验证，保存每折验证集表现最好的模型checkpoint。微调周期为5，文本长度截断为128。

分别使用`roberta`, `macbert`, `nezha`三个开源中文预训练模型进行post-pretrain，再各自5折微调，得到15个checkpoint，之后对这15个模型加权集成，最终在B榜Mean F1 Score为0.698138。具体细节见下表：

| 预训练模型 | MLM weight | ITM weight | MRFR weight | Mean F1 of 5 fold val | Ensemble weight |
| ---------- | ---------- | ---------- | ----------- | --------------------- | --------------- |
| roberta    | 0.5        | 0.3        | 0.2         | .6724                 | .1              |
|            |            |            |             | .6812                 |                 |
|            |            |            |             | .6846                 |                 |
|            |            |            |             | .6805                 |                 |
|            |            |            |             | .6846                 |                 |
| macbert    | 0.5        | 0.3        | 0.2         | .6866                 | .525            |
|            |            |            |             | .6850                 |                 |
|            |            |            |             | .6822                 |                 |
|            |            |            |             | .6808                 |                 |
|            |            |            |             | .6882                 |                 |
| nezha      | 0.6        | 0.1        | 0.3         | .6775                 | .375            |
|            |            |            |             | .6793                 |                 |
|            |            |            |             | .6815                 |                 |
|            |            |            |             | .6807                 |                 |
|            |            |            |             | .6841                 |                 |

# 代码运行流程

1、预训练`src/pretrain.py`

2、微调`src/kfold_finetune.py`

3、推断生成结果文件`src/kfold_inference.py`

具体运行命令见脚本。