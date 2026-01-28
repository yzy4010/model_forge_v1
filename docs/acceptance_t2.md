# T2.5 端到端验收清单与示例请求

> 目标：提供使用 **curl 或 Postman** 的端到端验收步骤，覆盖**成功与失败场景**。

## 0. 前置准备

- 服务已启动，假设基地址为：`http://localhost:8000`
- 所有示例以 curl 为主，Postman 可直接复用同样的请求参数。
- 所有路径示例中的 `<DATASET_ZIP>`、`<DATASET_ID>`、`<JOB_ID>` 请替换为真实值。

---

## A. 本地数据集注册验收

### A-1 成功场景：注册包含 train/valid/test + data.yaml 的目录

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/datasets/register_local" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smoking_v3",
    "root_dir": "<DATASET_ROOT_DIR>",
    "data_yaml": "data.yaml"
  }'
```

**期望**

- 响应包含 `dataset_id`。
- 服务器落盘存在以下文件：
  - `datasets/<id>/resolved_data.yaml`
  - `datasets/<id>/dataset_meta.json`

**验证示例**

```bash
# 将 <DATASET_ID> 替换为响应返回值
DATASET_ID=<DATASET_ID>
ls -l datasets/${DATASET_ID}/resolved_data.yaml
ls -l datasets/${DATASET_ID}/dataset_meta.json
```

### A-2 resolved_data.yaml 路径验证

**期望**

- `resolved_data.yaml` 中的路径为 **绝对路径**，且指向 `root_dir` 下的真实目录。
- 路径不包含 `../`。

**验证示例**

```bash
cat datasets/${DATASET_ID}/resolved_data.yaml
# 期望输出包含类似：
# train: /abs/path/.../train/images
# val:   /abs/path/.../valid/images 或 /abs/path/.../val/images
# test:  /abs/path/.../test/images（存在则写）
```

### A-3 错误场景：root_dir 不存在

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/datasets/register_local" \
  -H "Content-Type: application/json" \
  -d '{
    "root_dir": "/path/not/exist"
  }' -i
```

**期望**

- 返回 **HTTP 400**。
- 错误信息提示 `root_dir invalid`。

### A-4 错误场景：data.yaml 不存在

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/datasets/register_local" \
  -H "Content-Type: application/json" \
  -d '{
    "root_dir": "<DATASET_ROOT_DIR>",
    "data_yaml": "missing.yaml"
  }' -i
```

**期望**

- 返回 **HTTP 400**。
- 错误信息提示 `missing data.yaml`。

---

## B. 新训练验收（/train/yolo/new）

### B-1 训练启动

**请求（curl）**

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

- 响应返回 `job_id`。
- 目录 `outputs/<job_id>/...` 被创建。

**验证示例**

```bash
JOB_ID=<JOB_ID>
ls -l outputs/${JOB_ID}/
```

### B-2 训练中进度查询（epoch 级）

**请求（curl）**

```bash
curl -s "http://localhost:8000/train/${JOB_ID}/progress"
```

**期望**

- `epochs_done` 从 `0 -> 1 -> 2` 逐步变化。
- 当某个 epoch 完成后，`last` / `best_so_far` 字段出现非空值。

**建议**

- 每隔数秒轮询一次，直到 epoch 达到 2。

### B-3 完成结果查询

**请求（curl）**

```bash
curl -s "http://localhost:8000/train/${JOB_ID}/result"
```

**期望**

- `status = completed`。
- `best` 字段存在。
- `artifacts` 列表包含（存在则列出）：
  - `best.pt`
  - `last.pt`
  - `results.csv`
  - `args.yaml`
- `bundle.zip_path` 存在，且文件真实存在。

**验证示例**

```bash
# 假设 bundle.zip_path 返回在 JSON 的 bundle.zip_path 字段
BUNDLE_PATH=<BUNDLE_ZIP_PATH>
ls -l ${BUNDLE_PATH}
```

---

## C. 继续训练验收（/train/yolo/continue）

### C-1 finetune_best

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/continue" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "base_job_id": "<JOB_ID>",
    "strategy": "finetune_best",
    "params": {
      "epochs": 2
    }
  } '
```

**期望**

- 返回新的 `job_id`。
- 使用 base_job 的 `best.pt` 作为继续训练的初始化权重。
- 不覆盖 base_job 的输出目录。

### C-2 错误场景：base_job_id 不存在

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/continue" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "base_job_id": "job_not_exist",
    "strategy": "finetune_best",
    "params": {
      "epochs": 2
    }
  } ' -i
```

**期望**

- 返回 **HTTP 404 或 400**。
- 错误信息明确说明 base_job_id 不存在。

### C-3 错误场景：base_job 缺少 best.pt

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/continue" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "base_job_id": "<JOB_ID_WITHOUT_BEST>",
    "strategy": "finetune_best",
    "params": {
      "epochs": 2
    }
  } ' -i
```

**期望**

- 返回 **HTTP 400**。
- 错误信息明确说明 base_job 缺少 `best.pt`。

---

## D. 参数映射验收

### D-1 augment_level 不同档位可正常训练启动

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8s",
    "params": {
      "epochs": 2,
      "augment_level": "default"
    }
  } '
```

**期望**

- 返回 `job_id` 并正常启动训练。

### D-2 lr_scale 非法值校验

**请求（curl）**

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

**期望**

- 返回 **HTTP 400**。
- 错误信息明确说明 lr_scale 非法。

### D-3 device_policy=cuda 且无 CUDA

**请求（curl）**

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

**期望**

- 当环境无 CUDA 时返回 **HTTP 400**。
- 错误信息明确说明 CUDA 不可用。

### D-4 batch="auto" 在 CPU 环境默认 batch=2

**请求（curl）**

```bash
curl -s -X POST "http://localhost:8000/train/yolo/new" \
  -H "Content-Type: application/json" \
  -d ' {
    "dataset_id": "<DATASET_ID>",
    "model_spec": "yolov8n",
    "params": {
      "epochs": 2,
      "batch": "auto"
    }
  } '
```

**期望**

- 训练正常启动。
- 在 CPU 环境下，最终写入的 `args.yaml` 或训练 meta 中显示 `batch=2`。

**验证示例**

```bash
# 在 outputs/<job_id>/ 目录中寻找 args.yaml 或 meta 文件
cat outputs/${JOB_ID}/args.yaml
# 或
cat outputs/${JOB_ID}/meta.json
```
