# daguan-2019

## 简介
[达观文本智能信息抽取比赛](https://www.biendata.xyz/competition/datagrand/) code，比赛难点在于给出数据为加密文本，所以需要用给定语料pretrain预训练模型

## 模型
- 模型主要采用预训练模型+fineuning模式
- 预训练模型
   - 模型结构为6层的transformer encoder
   - 采用动态mask机制，不同epoch随机mask不同的字
   - loss设置为mask-language model loss
- 模型效果
  - 采用K-fold ensmble后，提交后F1为0.925

## 数据
- [数据下载](https://pan.baidu.com/s/1cYfK78cyAPncGDRk46hlGw),
提取码：r4dt
- 下载后，请把corpus.txt移至 NERData/mask_lm/ 下

## 预训练
- bash scripts/daguan/bash/run_dg_lm.sh
- [预训练模型下载](https://pan.baidu.com/s/1n_B_XQ0HijGbfEjqoK81GA 
),  提取码:th3e
- 下载后请移至 ckpt/daguan/ 下

## finetune
- bash scripts/daguan/bash/run_dg_lm.sh

## 模型导出
- bash scripts/daguan/bash/export.sh

## 框架使用说明
- [参考文档](https://opennmt.net/OpenNMT-tf/)

## 基于OpenNmt改进点
- 实现MaskedWordEmbedder，训练时可以实现token的动态mask, 见text_inputter.py
- 实现mask-language model loss， 见sequence_tagger.py make dense函数
- Catalog.py 新增模型类别TransformerSeqTagger，TransformerMaskLM
- 转移矩阵初始化支持，见sequence_tagger.py _call函数