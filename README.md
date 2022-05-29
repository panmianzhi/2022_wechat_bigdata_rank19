```shell
|-- config.py 
|-- data
|   |-- category_distribution.jpg
|   |-- category_id_map.py
|   |-- count_category.py # 统计物体类别数目
|   |-- data_helper.py # 加载微调数据
|   |-- lxrt_dataset.py # 加载预训练数据
|-- evaluate.py
|-- finetune.py # 微调
|-- inference.py # 生成最终结果文件代码
|-- lxrt_pretrain.sh
|-- models
|   |-- baseline_model.py 
|   |-- file_utils.py 
|   |-- finetune_model.py # 微调的模型
|   |-- lxrt.py # 双流模型，我们用不到
|   |-- optimization.py # 预训练阶段的优化器BertAdam
|   |-- uniter.py # 单流模型
|-- pretrain.py # 预训练
|-- run_finetune.sh # 微调预训脚本
|-- save # 保存checkpoint
|-- tricks
|   |-- EMA.py
|   |-- FGM.py
|-- util.py

```

