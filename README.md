# ModelForge v1

## 项目简介
ModelForge v1 是基于 ClearML 的统一训练平台（v1）。

## v1 目标
- API 驱动训练
- YOLO/UNet 可扩展
- ClearML offline → online

## 启动方式
1. 创建并激活虚拟环境（示例）
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 启动服务
   ```bash
   python -m uvicorn app.main:app --reload
   ```

4. 访问健康检查
   - http://127.0.0.1:8000/health

## 本地数据集注册

请求：`POST http://127.0.0.1:8000/datasets/register_local`

```json
{
  "name": "smoking_v3",
  "root_dir": "D:/projects/smoking.v3i.yolov8",
  "data_yaml": "data.yaml"
}
```

## Postman 示例（YOLO 训练）

请求：`POST http://127.0.0.1:8000/train/yolo/new`

```json
{
  "dataset_id": "<DATASET_ID>",
  "model_spec": "yolov8n",
  "params": {
    "epochs": 1,
    "batch": "auto",
    "imgsz": 640,
    "lr0": 0.01,
    "workers": 0,
    "patience": 5
  }
}
```
