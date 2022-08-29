# 复赛

唯一遗憾的是半监督（蒸馏）方案半途而废了。。。。。。

## 代码架构

```shell
|-- clip # clip代码仓库
|   |-- __init__.py
|   |-- bpe_simple_vocab_16e6.txt.gz
|   |-- clip.py
|   |-- model.py
|   |-- simple_tokenizer.py
|-- config.py # 参数配置
|-- data # dataloader相关代码
|   |-- category_id_map.py
|   |-- finetune_dataset.py # 加载微调数据
|   |-- pretrain_dataset.py # 加载预训练数据
|-- extract_features.py # 预先抽取clip图像特征保存本地
|-- finetune_swad.py # 微调
|-- models # 模型代码
|   |-- create_optimizer.py # 预训练优化器配置
|   |-- finetune_model_v2.py # 微调模型（单流）
|   |-- two_stream_model.py # 微调模型（双流）
|   |-- uniter.py # 骨干模型代码
|-- pretrain.py # 预训练
|-- tricks # 微调的tricks
|   |-- EMA.py
|   |-- FGM.py
|   |-- swad
|       |-- __pycache__
|       |   |-- __init__.cpython-36.pyc
|       |   |-- swa_utils.cpython-36.pyc
|       |   |-- swad.cpython-36.pyc
|       |-- swa_utils.py
|       |-- swad.py
|-- util.py
```

## 主要思路

方案思路很简单，仍是预训练-微调。

预训练采用了MLM、MFM、ITM三个任务，相关骨干代码参考了[2021QQ浏览器AIAC代码](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st)。

微调采用的主要trick包括fgm、[swad](https://github.com/khanrc/swad)两种。注意由于使用到了swad，学习率要根据原论文采用**恒定大学习率**。此外，微调模型分别构建了单流和双流两种，单流模型在post-pretrain之后的模型基础上微调，双流模型直接利用预训练模型macbert微调。

最后ensemble了如下三个模型，线上mean f1为0.7229

| model-arch | pretrain-ckpt | post-pretrain(yes/no) | ensemble weight | online mean f1 |
| ---------- | ------------- | --------------------- | --------------- | -------------- |
| single     | nezha         | yes                   | .35             | .710391        |
| single     | roberta       | yes                   | .35             | .713592        |
| dual       | macbert       | no                    | .3              | w/o test       |

## 补充

赛后讨论代码：https://developers.weixin.qq.com/community/develop/article/doc/0006626c9e8ca0fedb5ea543856413

