import cv2
import json
import os
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai

# ==========================================
# 1. 视觉感知 Agent (Visual Perception Agent)
# ==========================================
class VisualPerceptionAgent:
    def __init__(self, model_path='yolov8n.pt'):
        # 加载 YOLOv8 模型 (可替换为你自己训练的路景分割/检测权重)
        print("[System] 初始化视觉感知 Agent...")
        self.model = YOLO(model_path)
        # 定义置信度阈值：低于此阈值认为是"异常/长尾"目标
        self.low_conf_threshold = 0.60 

    def process(self, image_path):
        print(f"[Visual Agent] 正在处理图像: {image_path}")
        img = cv2.imread(image_path)
        results = self.model(img, verbose=False)[0]
        
        standard_objects = []
        anomaly_regions = []

        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            crop_img = img[y1:y2, x1:x2]
            
            obj_data = {
                "bbox": [x1, y1, x2, y2],
                "class": name,
                "confidence": round(conf, 3)
            }

            # 核心逻辑：分流处理
            if conf >= self.low_conf_threshold:
                standard_objects.append(obj_data)
            else:
                # 提取低置信度区域的图像，传递给下游 Agent
                anomaly_regions.append({
                    "data": obj_data,
                    "crop": crop_img
                })
                
        return standard_objects, anomaly_regions

# ==========================================
# 2. 深度推理 Agent (Deep Reasoning Agent)
# ==========================================
class DeepReasoningAgent:
    def __init__(self, api_key):
        print("[System] 初始化深度推理 Agent...")
        genai.configure(api_key=api_key)
        # 使用多模态大模型
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, crop_img_cv2, base_class):
        print(f"[Reasoning Agent] 触发长链推理，分析低置信度目标 (初步判断为: {base_class})...")
        
        # OpenCV BGR 转 PIL RGB
        color_coverted = cv2.cvtColor(crop_img_cv2, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        
        prompt = f"""
        你是一个自动驾驶复杂路况异常分析专家。
        前置视觉模型对这张截取的障碍物图像给出的初步判断是 '{base_class}'，但置信度很低。
        请执行长链推理：
        1. 仔细观察图像特征（形状、纹理、材质、环境上下文）。
        2. 判断这到底是什么？（例如：是破损的交通锥、散落的纸箱、异形车辆，还是恶劣天气引起的伪影？）
        3. 给出该障碍物对自动驾驶车辆的潜在危险等级（低/中/高）。
        
        请严格以JSON格式输出，不要包含其他解释文本。格式如下：
        {{"refined_class": "详细类别描述", "reasoning": "你的推理过程", "danger_level": "高/中/低"}}
        """
        
        try:
            response = self.model.generate_content([prompt, pil_image])
            # 清理 Markdown 标记以提取 JSON
            result_text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(result_text)
        except Exception as e:
            print(f"[Reasoning Agent] 推理失败: {e}")
            return {"refined_class": "unknown", "reasoning": "API Error", "danger_level": "unknown"}

# ==========================================
# 3. 校验与标注 Agent (Validation Agent)
# ==========================================
class ValidationAgent:
    def __init__(self):
        print("[System] 初始化校验与标注 Agent...")

    def generate_annotation(self, image_id, standard_objs, reasoned_anomalies):
        print("[Validation Agent] 正在生成结构化标注数据...")
        
        final_annotation = {
            "image_id": image_id,
            "metadata": {
                "total_objects": len(standard_objs) + len(reasoned_anomalies),
                "anomalies_resolved": len(reasoned_anomalies)
            },
            "standard_objects": standard_objs,
            "complex_anomalies": reasoned_anomalies
        }
        
        # 保存为标准 JSON 文件
        output_file = f"annotation_{image_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_annotation, f, ensure_ascii=False, indent=4)
            
        print(f"[Validation Agent] 闭环完成！标注数据已保存至 {output_file}\n")
        return final_annotation

# ==========================================
# 4. 主控逻辑流 (Orchestrator)
# ==========================================
def main():
    # 替换为你的大模型 API 密钥 (这里留空的话，推理节点会报错但系统能跑通)
    API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE") 
    TEST_IMAGE = "test_road.jpg" # 准备一张包含路况的测试图片
    
    if not os.path.exists(TEST_IMAGE):
        print(f"错误: 找不到测试图片 {TEST_IMAGE}。请放置一张图片在同级目录下。")
        # 生成一张纯色图片用于防崩测试
        import numpy as np
        cv2.imwrite(TEST_IMAGE, np.zeros((640, 640, 3), dtype=np.uint8))
        print(f"已自动生成空白 {TEST_IMAGE} 用于测试，请替换为真实路景图以获得真实效果。")

    # 1. 实例化多 Agent 团队
    vision_agent = VisualPerceptionAgent()
    reasoning_agent = DeepReasoningAgent(api_key=API_KEY)
    validation_agent = ValidationAgent()

    # 2. 运行核心业务流
    print("\n" + "="*40)
    print("开始执行多 Agent 协同路景分析流")
    print("="*40)

    # Step 1: 视觉初筛
    standard_objs, anomaly_regions = vision_agent.process(TEST_IMAGE)
    print(f"[System] 视觉感知完成: 发现 {len(standard_objs)} 个标准目标, {len(anomaly_regions)} 个低置信度异常区域。")

    # Step 2: 触发深度推理
    reasoned_anomalies = []
    for idx, anomaly in enumerate(anomaly_regions):
        print(f"\n--- 处理异常区域 {idx+1}/{len(anomaly_regions)} ---")
        base_class = anomaly["data"]["class"]
        crop_img = anomaly["crop"]
        
        # LLM 介入分析
        llm_analysis = reasoning_agent.analyze(crop_img, base_class)
        
        # 合并数据
        enriched_data = anomaly["data"].copy()
        enriched_data.update({
            "llm_analysis": llm_analysis,
            "is_reviewed_by_llm": True
        })
        reasoned_anomalies.append(enriched_data)

    # Step 3: 校验与结构化输出
    print("\n" + "="*40)
    final_result = validation_agent.generate_annotation(
        image_id=TEST_IMAGE.split('.')[0],
        standard_objs=standard_objs,
        reasoned_anomalies=reasoned_anomalies
    )

    # 打印最终效果
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()