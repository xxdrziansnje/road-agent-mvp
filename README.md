# 🚦 Complex Road Scene Anomaly Analysis Agent 

基于 YOLOv8 与大语言模型（LLM）协同的复杂路景异常分析与标注 Agent 系统 (MVP)。

## 🎯 核心痛点与解决思路
传统路景语义分割在遇到长尾异常情况（如极端天气、未见过的非标准障碍物）时极易误判，且人工复核自动驾驶数据集的成本极高。

本项目采用**多 Agent 协同架构**：
1. **视觉感知 Agent (YOLOv8)**：低成本进行实时帧处理和基础目标检测。
2. **深度推理 Agent (Multimodal LLM)**：当视觉模型遇到**低置信度 (<0.60)** 区域时被触发。通过长链推理分析复杂障碍物属性及潜在危险等级。
3. **校验 Agent**：自动生成包含推理依据的结构化 JSON 标注数据。

## 🚀 快速开始

### 1. 环境依赖
```bash
pip install ultralytics opencv-python google-generativeai pillow
```

### 2. 运行测试
请确保配置了有效的大模型 API Key，准备一张测试图片放置于根目录，然后运行核心主程序：
```bash
python code.py
```

## 📊 核心业务流价值
本项目通过前端低成本视觉模型过滤，按需触发高成本多模态大模型长链推理，极大地平衡了工程落地成本（Token 消耗和推理延迟）与长尾路况的识别准确率。
