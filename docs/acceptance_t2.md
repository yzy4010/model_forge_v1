# T2.6 训练闭环验收（YOLO new/continue + progress/result）

> 目标：提供 **手动验收步骤**（Swagger / Postman / curl 均可），覆盖成功路径与关键失败路径。

## 0. 前置准备

- 服务已启动：`python -m uvicorn app.main:app --reload`
- 基地址假设为：`http://localhost:8000`
- 先完成 `register_local` 拿到 `dataset_id`（参考 `docs/acceptance_t2_0R.md`）。

---

## A. 新模型训练 /train/yolo/new（from_pretrained）

### A-1 启动训练（成功路径）

**请求**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8n",
    "params": {
      "epochs": 2
    }
  } '
```

**期望**

- 返回 `job_id`。
- 训练模式为 **from_pretrained**。
- 目录 `outputs/<job_id>/` 被创建。

### A-2 轮询进度 /train/{job_id}/progress

**请求**

```bash
curl -s "http://localhost:8000/train/<JOB_ID>/progress"
```

**期望**

- `epochs_done` 随训练递增（例如 0 → 1 → 2）。
- `last` / `best_so_far` 在 epoch 产生后非空。

### A-3 结果获取 /train/{job_id}/result

**请求**

```bash
curl -s "http://localhost:8000/train/<JOB_ID>/result"
```

**期望**

- `status = completed`。
- `best` 有值。
- `bundle.zip_path` 存在，且文件真实存在。

**验证示例**

```bash
BUNDLE_PATH=<BUNDLE_ZIP_PATH>
ls -l "${BUNDLE_PATH}"
```

---

## B. 继续训练 /train/yolo/continue（finetune_best）

### B-1 启动继续训练（成功路径）

> 必须基于已有 `base_job_id` 的 `best.pt` 继续训练，且 **不覆盖 base_job** 的输出目录。

**请求**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/continue" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "base_job_id": "<BASE_JOB_ID>",
    "continue_strategy": "finetune_best",
    "params": {
      "epochs": 2
    }
  } '
```

**期望**

- 返回新的 `job_id`。
- 训练模式为 **finetune_best**（基于 `best.pt`）。
- `outputs/<job_id>/` 为新目录，`outputs/<BASE_JOB_ID>/` 保持不变。
- `result` 中 `bundle.zip_path` 存在。

---

## C. 关键失败路径

### C-1 augment_level 非法

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8n",
    "params": {
      "epochs": 2,
      "augment_level": "not_supported"
    }
  } ' -i
```

**期望**：HTTP 400，提示 augment_level 不支持。

### C-2 lr_scale 非法

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8n",
    "params": {
      "epochs": 2,
      "lr_scale": 1.3
    }
  } ' -i
```

**期望**：HTTP 400，提示 lr_scale 不支持。

### C-3 device_policy=cuda 且无 GPU

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8n",
    "params": {
      "epochs": 2,
      "device_policy": "cuda"
    }
  } ' -i
```

**期望**：无 CUDA 时返回 HTTP 400，提示 `cuda not available`。
