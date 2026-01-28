# ModelForge v1

## 项目简介

ModelForge v1：提供 **本地数据集注册** + **YOLO 训练** 的轻量服务端，面向前端/手动调用即可完成训练闭环。

- 接口对接文档：[`docs/api_integration.md`](docs/api_integration.md)

---

## 环境要求

- Windows 10/11（推荐）
- Python 3.9
- Ultralytics 8.4.7（已在依赖中锁定）
- GPU/CPU 自动切换（`device_policy=auto`）

---

## 安装步骤（10 分钟跑通）

### 1) 创建 venv

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 启动服务

```bash
python -m uvicorn app.main:app --reload
```

### 4) 打开 Swagger

- http://127.0.0.1:8000/docs

---

## 支持的 model_spec

- yolov8n / yolov8s / yolov8m

---

## 最小跑通（复制即可）

### 1) register_local

请求：`POST /datasets/register_local`

```json
{
  "name": "smoking_v3",
  "root_dir": "D:/datasets/smoking.v3i.yolov8",
  "data_yaml": "data.yaml"
}
```

> Windows 路径写法：推荐 `D:/...` 或 `D:\\...`，避免 `JSON Invalid \escape`。

### 2) train/yolo/new（epochs=2）

请求：`POST /train/yolo/new`

```json
{
  "dataset_id": "<DATASET_ID>",
  "model_spec": "yolov8n",
  "params": {
    "epochs": 2
  }
}
```

### 3) progress 轮询

请求：`GET /train/{job_id}/progress`

```bash
curl -s "http://127.0.0.1:8000/train/<JOB_ID>/progress"
```

### 4) result 获取（含 bundle.zip_path）

请求：`GET /train/{job_id}/result`

```bash
curl -s "http://127.0.0.1:8000/train/<JOB_ID>/result"
```

响应中包含：

- `bundle.zip_path`（训练产物 zip）
- `artifacts`（best.pt / last.pt / results.csv / args.yaml）

### 5) train/yolo/continue（基于 base_job_id）

请求：`POST /train/yolo/continue`

```json
{
  "dataset_id": "<DATASET_ID>",
  "base_job_id": "<BASE_JOB_ID>",
  "continue_strategy": "finetune_best",
  "params": {
    "epochs": 2
  }
}
```

---

## 接口冻结清单（对外业务接口）

数据集：
- `POST /datasets/register_local`
- `GET /datasets/{dataset_id}`

训练：
- `POST /train/yolo/new`
- `POST /train/yolo/continue`

查询：
- `GET /train/{job_id}/progress`
- `GET /train/{job_id}/result`

运维：
- `GET /health`
- `GET /`（可选）

---

## 输出产物目录说明

- `outputs/<job_id>/artifacts/`：训练产物（best.pt / last.pt / results.csv / args.yaml）
- `outputs/<job_id>/bundle/model_forge_<job_id>.zip`：训练结果打包（`bundle.zip_path`）

---

## 常见错误排查

- **JSON Invalid \escape**：Windows 路径不要用单个 `\`，改用 `D:/...` 或 `D:\\...`。
- **Python 3.9 不支持 `str | None`**：请使用 `Optional[str]` 语法。
- **device_policy=cuda 无 GPU**：环境不支持 CUDA，请改用 `device_policy=auto` 或 `cpu`。
- **dataset 结构不符合**：需要 `train/images` + `train/labels`，以及 `valid/images` + `valid/labels`（或 `val/`）。
