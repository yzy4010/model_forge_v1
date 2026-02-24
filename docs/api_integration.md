# 接口对接文档（ModelForge v1）

## 概述

- **项目目标**：ModelForge v1（本地数据集注册 + YOLO 训练）。
- **设计约束**：不上传数据集，使用本机/服务器可访问的绝对路径或相对路径（相对 `root_dir`）。
- **响应约定**：所有接口返回 JSON；错误返回 FastAPI 标准结构 `{ "detail": "..." }`。

## 全局约定

- **Windows 路径 JSON 转义规则**：必须使用 `D:/...` 或 `D:\\...`，不要用单个 `\`，否则会触发 `JSON Invalid \escape`。
- **训练任务模型**：
  - **new**：`from_pretrained`（从预训练权重启动训练）。
  - **continue**：`finetune_best`（基于 `base_job_id` 的 `best.pt` 微调，输出到新 job）。
- **artifacts 与 bundle.zip 的含义**：
  - `artifacts`：训练产物文件列表（如 `best.pt` / `last.pt` / `results.csv` / `args.yaml`）。
  - `bundle.zip_path`：平台交付包路径，打包 `artifacts` 和 `meta.json`，用于交付/下载。

## 接口列表

> 仅包含 9 个业务接口：

- `POST /datasets/register_local`
- `GET /datasets/{dataset_id}`
- `POST /train/yolo/new`
- `POST /train/yolo/continue`
- `GET /train/{job_id}/progress`
- `GET /train/{job_id}/result`
- `POST /infer/stream`
- `POST /infer/{job_id}/stop`
- `GET /preview/{job_id}`

---

## POST /datasets/register_local

**功能说明**：注册本地 YOLO 数据集，并生成 `resolved_data.yaml`。

### Request

- **Content-Type**：`application/json`

**请求体 JSON 示例**

```json
{
  "name": "smoking_v3",
  "root_dir": "D:/datasets/smoking.v3i.yolov8",
  "data_yaml": "data.yaml"
}
```

**逐字段注解**

- `name`
  - 含义：数据集名称，仅用于展示与元数据记录。
  - 是否必填：否。
  - 取值范围/枚举：任意字符串。
  - 默认行为：省略时为 `null`。
  - 常见错误：无特定错误。
- `root_dir`
  - 含义：本地 YOLO 数据集根目录（包含 `train/`、`valid/` 或 `val/`、`test/` 与 `data.yaml`）。
  - 是否必填：是。
  - 取值范围/枚举：合法路径字符串（本机/服务器可访问）。
  - 默认行为：无。
  - 常见错误：`root_dir invalid`（路径不存在或非目录）。
- `data_yaml`
  - 含义：`data.yaml` 路径，可为相对 `root_dir` 的相对路径或绝对路径。
  - 是否必填：否。
  - 取值范围/枚举：合法路径字符串。
  - 默认行为：省略时默认使用 `root_dir/data.yaml`。
  - 常见错误：`missing data.yaml`（路径不存在）。

### Response

**成功返回 JSON 示例**

```json
{
  "dataset_id": "b2a1d0f6f2c744b78a95e9c4f1d0c321",
  "name": "smoking_v3",
  "root_dir": "D:/datasets/smoking.v3i.yolov8",
  "resolved_data_yaml_path": "D:/projects/model_forge_v1/datasets/b2a1d0f6f2c744b78a95e9c4f1d0c321/resolved_data.yaml",
  "stats": {
    "nc": 2,
    "names": ["smoke", "person"],
    "train_images": 1200,
    "valid_images": 300,
    "test_images": 0,
    "train_labels": 1200,
    "valid_labels": 300,
    "test_labels": 0
  },
  "created_at": "2024-05-10T12:34:56Z"
}
```

**逐字段注解**

- `dataset_id`
  - 含义：数据集注册后的唯一 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：成功注册后返回。
- `name`
  - 含义：数据集名称。
  - 取值：字符串或 `null`。
  - 出现时机：成功注册后返回。
- `root_dir`
  - 含义：解析后的数据集根目录（绝对路径）。
  - 取值：路径字符串。
  - 出现时机：成功注册后返回。
- `resolved_data_yaml_path`
  - 含义：系统生成的 `resolved_data.yaml` 绝对路径。
  - 取值：路径字符串。
  - 出现时机：成功注册后返回。
- `stats`
  - 含义：数据集统计信息对象。
  - 取值：对象，推荐包含 key：`nc`、`names`、`train_images`、`valid_images`、`test_images`、`train_labels`、`valid_labels`、`test_labels`。
  - 说明：Swagger 中可能出现 `additionalProp1`，那是泛型占位，不是实际业务字段。
  - 出现时机：成功注册后返回。
- `created_at`
  - 含义：注册时间（UTC）。
  - 取值：ISO 8601 字符串。
  - 出现时机：成功注册后返回。

**典型调用流程中的位置**：注册数据集 → 获取 `dataset_id` → 训练 `new` / `continue`。

**常见错误与排查**

1. `root_dir invalid`：确认路径存在且为目录。
2. `missing data.yaml`：确认 `data_yaml` 路径正确，或 `root_dir/data.yaml` 存在。

---

## GET /datasets/{dataset_id}

**功能说明**：查询已注册数据集信息。

### Request

- **Content-Type**：无（GET 请求）

### Response

**成功返回 JSON 示例**

```json
{
  "dataset_id": "b2a1d0f6f2c744b78a95e9c4f1d0c321",
  "name": "smoking_v3",
  "root_dir": "D:/datasets/smoking.v3i.yolov8",
  "resolved_data_yaml_path": "D:/projects/model_forge_v1/datasets/b2a1d0f6f2c744b78a95e9c4f1d0c321/resolved_data.yaml",
  "stats": {
    "nc": 2,
    "names": ["smoke", "person"],
    "train_images": 1200,
    "valid_images": 300,
    "test_images": 0,
    "train_labels": 1200,
    "valid_labels": 300,
    "test_labels": 0
  },
  "created_at": "2024-05-10T12:34:56Z"
}
```

**逐字段注解**

- `dataset_id`
  - 含义：数据集唯一 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：查询成功返回。
- `name`
  - 含义：数据集名称。
  - 取值：字符串或 `null`。
  - 出现时机：查询成功返回。
- `root_dir`
  - 含义：数据集根目录（绝对路径）。
  - 取值：路径字符串。
  - 出现时机：查询成功返回。
- `resolved_data_yaml_path`
  - 含义：解析后的 `resolved_data.yaml` 绝对路径。
  - 取值：路径字符串。
  - 出现时机：查询成功返回。
- `stats`
  - 含义：数据集统计信息对象。
  - 取值：对象，推荐包含 key：`nc`、`names`、`train_images`、`valid_images`、`test_images`、`train_labels`、`valid_labels`、`test_labels`。
  - 说明：Swagger 中可能出现 `additionalProp1`，那是泛型占位，不是实际业务字段。
  - 出现时机：查询成功返回。
- `created_at`
  - 含义：注册时间（UTC）。
  - 取值：ISO 8601 字符串。
  - 出现时机：查询成功返回。

**典型调用流程中的位置**：注册后用于查询确认数据集元信息。

**常见错误与排查**

1. `dataset_id not found`：确认 `dataset_id` 是否存在。
2. `dataset_meta.json is invalid`：数据集元数据损坏，需重新注册。

---

## POST /train/yolo/new

**功能说明**：启动新的 YOLO 训练任务（from_pretrained）。

### Request

- **Content-Type**：`application/json`

**请求体 JSON 示例**

```json
{
  "dataset_id": "b2a1d0f6f2c744b78a95e9c4f1d0c321",
  "model_spec": "yolov8n",
  "params": {
    "epochs": 50,
    "imgsz": 640,
    "batch": "auto",
    "patience": 10,
    "augment_level": "default",
    "lr_scale": 1.0,
    "device_policy": "auto",
    "seed": 42,
    "val": true,
    "save": true
  }
}
```

**逐字段注解**

- `dataset_id`
  - 含义：数据集 ID。
  - 是否必填：是。
  - 取值范围/枚举：32 位 hex 字符串。
  - 默认行为：无。
  - 常见错误：`dataset_id not found`（数据集不存在）。
- `model_spec`
  - 含义：模型规格（YOLOv8 系列），决定训练起点权重与算力消耗。
  - 是否必填：是。
  - 取值范围/枚举：`yolov8n` / `yolov8s` / `yolov8m`。
  - 默认行为：无（必须显式指定）。
  - 说明：
    - `yolov8n`：最轻最快，适合 CPU/快速验证，精度较低。
    - `yolov8s`：更均衡，速度快且精度明显提升，常用默认选择。
    - `yolov8m`：精度更高，训练更慢、资源需求更大，适合追求更好效果的场景（建议 GPU）。
  - 常见错误：
    - `Unsupported model_spec: xxx. Allowed: yolov8n, yolov8s, yolov8m`（不在枚举内）。
    - `Failed to download yolov8m.pt...`（无网络首次下载失败 → 将 `yolov8m.pt` 放到权重目录）。
    - `CUDA out of memory`（GPU 不可用或 batch/imgsz 过大导致 OOM → 降低 batch/imgsz 或改用 `yolov8s`/`yolov8n`）。
    - `...not found`（权重文件缺失/路径错误 → 检查权重目录与文件名）。
- `params`
  - 含义：训练参数对象。
  - 是否必填：是。
  - 取值范围/枚举：见下。
  - 默认行为：缺省字段使用默认值。
  - 常见错误：见各字段。

**params 内逐字段注解**

- `epochs`
  - 含义：训练轮数。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 50。
  - 常见错误：非数字导致训练失败。
- `imgsz`
  - 含义：训练图片尺寸。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 640。
  - 常见错误：过小/过大导致训练异常。
- `batch`
  - 含义：batch 大小。
  - 是否必填：否。
  - 取值范围/枚举：正整数或 `"auto"`。
  - 默认行为：默认 `"auto"`（CUDA=8，CPU=2）。
  - 常见错误：`unsupported batch value` / `batch must be positive`。
- `patience`
  - 含义：早停耐心值。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 10。
  - 常见错误：负值导致训练失败。
- `augment_level`
  - 含义：数据增强等级。
  - 是否必填：否。
  - 取值范围/枚举：`default` 等预设。
  - 默认行为：默认 `default`。
  - 常见错误：不支持值会返回 400。
- `lr_scale`
  - 含义：学习率缩放倍数。
  - 是否必填：否。
  - 取值范围/枚举：`0.5` / `1.0` / `2.0`。
  - 默认行为：默认 `1.0`。
  - 常见错误：`unsupported lr_scale`。
- `device_policy`
  - 含义：设备策略。
  - 是否必填：否。
  - 取值范围/枚举：`auto` / `cpu` / `cuda`。
  - 默认行为：默认 `auto`。
  - 常见错误：`unsupported device_policy` / `cuda not available`。
- `seed`
  - 含义：随机种子。
  - 是否必填：否。
  - 取值范围/枚举：整数或 `null`。
  - 默认行为：默认 `null`。
  - 常见错误：无特定错误。
- `val`
  - 含义：是否在训练过程中执行验证。
  - 是否必填：否。
  - 取值范围/枚举：`true` / `false`。
  - 默认行为：默认 `true`。
  - 常见错误：无特定错误。
- `save`
  - 含义：是否保存模型权重。
  - 是否必填：否。
  - 取值范围/枚举：`true` / `false`。
  - 默认行为：默认 `true`。
  - 常见错误：无特定错误。

### Response

**成功返回 JSON 示例**

```json
{
  "job_id": "8f7b6c9d3c2a4e4aa0b1f91234567890",
  "status": "queued"
}
```

**逐字段注解**

- `job_id`
  - 含义：训练任务 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：创建任务后返回。
- `status`
  - 含义：训练任务状态。
  - 取值/枚举：`queued` / `running` / `completed` / `failed`。
  - 出现时机：创建任务后返回（初始为 `queued`）。

**典型调用流程中的位置**：注册数据集 → new → progress 轮询 → result。

**常见错误与排查**

1. `Unsupported model_spec: xxx. Allowed: yolov8n, yolov8s, yolov8m`：检查 `model_spec` 是否在枚举内。
2. `Failed to download yolov8m.pt...`：离线环境首次下载失败时，将 `yolov8m.pt` 放到权重目录并重试。
3. `CUDA out of memory`：GPU 不可用或 batch/imgsz 过大导致 OOM，建议降低 batch/imgsz 或改用 `yolov8s`/`yolov8n`。
4. `...not found`：权重文件缺失或路径错误，检查权重目录与文件名。
5. `unsupported lr_scale` / `unsupported device_policy`：检查参数是否在允许范围内。

---

## POST /train/yolo/continue

**功能说明**：基于已完成任务的 `best.pt` 继续训练（finetune_best）。

### Request

- **Content-Type**：`application/json`

**请求体 JSON 示例**

```json
{
  "dataset_id": "b2a1d0f6f2c744b78a95e9c4f1d0c321",
  "base_job_id": "8f7b6c9d3c2a4e4aa0b1f91234567890",
  "continue_strategy": "finetune_best",
  "params": {
    "epochs": 20,
    "imgsz": 640,
    "batch": "auto",
    "patience": 10,
    "augment_level": "default",
    "lr_scale": 1.0,
    "freeze": 0,
    "device_policy": "auto",
    "seed": 42,
    "val": true,
    "save": true
  }
}
```

**逐字段注解**

- `dataset_id`
  - 含义：数据集 ID。
  - 是否必填：是。
  - 取值范围/枚举：32 位 hex 字符串。
  - 默认行为：无。
  - 常见错误：`dataset_id not found`（数据集不存在）。
- `base_job_id`
  - 含义：基准训练任务 ID，必须已完成且存在 `best.pt`。
  - 是否必填：是。
  - 取值范围/枚举：32 位 hex 字符串。
  - 默认行为：无。
  - 常见错误：`base_job_id not found` / `best.pt not found for base_job_id`。
- `continue_strategy`
  - 含义：继续训练策略。
  - 是否必填：是。
  - 取值范围/枚举：`finetune_best`（默认）；`resume_last`（如未实现将返回错误）。
  - 默认行为：默认 `finetune_best`。
  - 常见错误：`resume_last not supported for base_job_id`。
- `params`
  - 含义：训练参数对象。
  - 是否必填：是。
  - 取值范围/枚举：见下。
  - 默认行为：缺省字段使用默认值。
  - 常见错误：见各字段。

**params 内逐字段注解**

- `epochs`
  - 含义：训练轮数。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 20。
  - 常见错误：非数字导致训练失败。
- `imgsz`
  - 含义：训练图片尺寸。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 640。
  - 常见错误：过小/过大导致训练异常。
- `batch`
  - 含义：batch 大小。
  - 是否必填：否。
  - 取值范围/枚举：正整数或 `"auto"`。
  - 默认行为：默认 `"auto"`（CUDA=8，CPU=2）。
  - 常见错误：`unsupported batch value` / `batch must be positive`。
- `patience`
  - 含义：早停耐心值。
  - 是否必填：否。
  - 取值范围/枚举：正整数。
  - 默认行为：默认 10。
  - 常见错误：负值导致训练失败。
- `augment_level`
  - 含义：数据增强等级。
  - 是否必填：否。
  - 取值范围/枚举：`default` 等预设。
  - 默认行为：默认 `default`。
  - 常见错误：不支持值会返回 400。
- `lr_scale`
  - 含义：学习率缩放倍数。
  - 是否必填：否。
  - 取值范围/枚举：`0.5` / `1.0` / `2.0`。
  - 默认行为：默认 `1.0`。
  - 常见错误：`unsupported lr_scale`。
- `freeze`
  - 含义：冻结层数（仅部分版本支持）。
  - 是否必填：否。
  - 取值范围/枚举：非负整数。
  - 默认行为：默认 0（不冻结）。
  - 常见错误：当前版本不支持时会在 `warnings` 中提示。
- `device_policy`
  - 含义：设备策略。
  - 是否必填：否。
  - 取值范围/枚举：`auto` / `cpu` / `cuda`。
  - 默认行为：默认 `auto`。
  - 常见错误：`unsupported device_policy` / `cuda not available`。
- `seed`
  - 含义：随机种子。
  - 是否必填：否。
  - 取值范围/枚举：整数或 `null`。
  - 默认行为：默认 `null`。
  - 常见错误：无特定错误。
- `val`
  - 含义：是否在训练过程中执行验证。
  - 是否必填：否。
  - 取值范围/枚举：`true` / `false`。
  - 默认行为：默认 `true`。
  - 常见错误：无特定错误。
- `save`
  - 含义：是否保存模型权重。
  - 是否必填：否。
  - 取值范围/枚举：`true` / `false`。
  - 默认行为：默认 `true`。
  - 常见错误：无特定错误。

### Response

**成功返回 JSON 示例**

```json
{
  "job_id": "4c1a9b2e3d4f5a6b7c8d901234567890",
  "status": "queued"
}
```

**逐字段注解**

- `job_id`
  - 含义：训练任务 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：创建任务后返回。
- `status`
  - 含义：训练任务状态。
  - 取值/枚举：`queued` / `running` / `completed` / `failed`。
  - 出现时机：创建任务后返回（初始为 `queued`）。

**典型调用流程中的位置**：注册数据集 → new 完成 → continue → progress 轮询 → result。

**常见错误与排查**

1. `base_job_id not found`：确认 `base_job_id` 是否存在。
2. `best.pt not found for base_job_id`：基准训练未产出 `best.pt`。

---

## GET /train/{job_id}/progress

**功能说明**：查询训练进度与最近指标（用于前端展示）。

### Request

- **Content-Type**：无（GET 请求）

### Response

**成功返回 JSON 示例（训练中）**

```json
{
  "job_id": "8f7b6c9d3c2a4e4aa0b1f91234567890",
  "status": "running",
  "epochs_total": 50,
  "epochs_done": 2,
  "last": {
    "epoch": 2,
    "box_loss": 1.23,
    "cls_loss": 0.45,
    "dfl_loss": 0.98,
    "precision": 0.71,
    "recall": 0.68,
    "map50": 0.62,
    "map50_95": 0.41
  },
  "best_so_far": {
    "epoch": 2,
    "box_loss": 1.23,
    "cls_loss": 0.45,
    "dfl_loss": 0.98,
    "precision": 0.71,
    "recall": 0.68,
    "map50": 0.62,
    "map50_95": 0.41
  },
  "history_tail": [
    {
      "epoch": 1,
      "box_loss": 1.35,
      "cls_loss": 0.49,
      "dfl_loss": 1.02,
      "precision": 0.62,
      "recall": 0.59,
      "map50": 0.55,
      "map50_95": 0.35
    },
    {
      "epoch": 2,
      "box_loss": 1.23,
      "cls_loss": 0.45,
      "dfl_loss": 0.98,
      "precision": 0.71,
      "recall": 0.68,
      "map50": 0.62,
      "map50_95": 0.41
    }
  ]
}
```

**逐字段注解**

- `job_id`
  - 含义：训练任务 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：查询成功返回。
- `status`
  - 含义：训练任务状态。
  - 取值/枚举：`running` / `completed` / `failed`。
  - 出现时机：查询成功返回。
- `epochs_total`
  - 含义：总训练轮数，可能为 `null`（元数据缺失时）。
  - 取值：整数或 `null`。
  - 出现时机：查询成功返回。
- `epochs_done`
  - 含义：已完成轮数。
  - 取值：非负整数。
  - 出现时机：查询成功返回。
- `last`
  - 含义：最新一轮的指标对象；若暂无结果则为 `null`。
  - 取值：对象或 `null`。
  - 出现时机：results.csv 可解析时。
- `best_so_far`
  - 含义：目前为止最佳指标（以 `map50` 最大为准）。
  - 取值：对象或 `null`。
  - 出现时机：results.csv 可解析时。
- `history_tail`
  - 含义：最近 N 个 epoch 的指标数组，用于 UI 展示，不用于最终交付。
  - 取值：数组。
  - 出现时机：results.csv 可解析时。

**last/best_so_far/history_tail 指标字段说明**

- `epoch`：epoch 序号。
- `box_loss`：box 回归损失。
- `cls_loss`：分类损失。
- `dfl_loss`：分布式回归损失。
- `precision`：精度。
- `recall`：召回率。
- `map50`：mAP@0.5。
- `map50_95`：mAP@0.5:0.95。

**典型调用流程中的位置**：new/continue 启动后轮询进度，直到完成。

**常见错误与排查**

1. `job_id not found`：确认 `job_id` 是否存在。
2. `training not finished`：需等待训练结束再调用 `/result`。

---

## GET /train/{job_id}/result

**功能说明**：获取训练完成后的结果与交付包路径。

### Request

- **Content-Type**：无（GET 请求）

### Response

**成功返回 JSON 示例**

```json
{
  "job_id": "8f7b6c9d3c2a4e4aa0b1f91234567890",
  "status": "completed",
  "train_mode": "from_pretrained",
  "dataset_id": "b2a1d0f6f2c744b78a95e9c4f1d0c321",
  "model_name_or_path": "yolov8n.pt",
  "hyperparams": {
    "epochs": 50,
    "imgsz": 640,
    "batch": 8,
    "patience": 10,
    "augment_level": "default",
    "lr_scale": 1.0,
    "device_policy": "auto",
    "seed": 42,
    "val": true,
    "save": true,
    "lr0": 0.01,
    "workers": 0
  },
  "epochs_total": 50,
  "epochs_done": 50,
  "best": {
    "epoch": 48,
    "map50": 0.72,
    "map50_95": 0.48,
    "precision": 0.78,
    "recall": 0.70
  },
  "artifacts": ["best.pt", "last.pt", "results.csv", "args.yaml"],
  "bundle": {
    "zip_path": "D:/projects/model_forge_v1/outputs/8f7b6c9d3c2a4e4aa0b1f91234567890/bundle/model_forge_8f7b6c9d3c2a4e4aa0b1f91234567890.zip"
  },
  "created_at": "2024-05-10T12:34:56Z",
  "finished_at": "2024-05-10T12:45:12Z"
}
```

**逐字段注解**

- `job_id`
  - 含义：训练任务 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：训练完成后返回。
- `status`
  - 含义：训练任务状态。
  - 取值/枚举：`completed`（成功时）。
  - 出现时机：训练完成后返回。
- `train_mode`
  - 含义：训练模式。
  - 取值/枚举：`from_pretrained` / `finetune` / `resume`。
  - 出现时机：训练完成后返回。
- `dataset_id`
  - 含义：数据集 ID。
  - 取值：32 位 hex 字符串。
  - 出现时机：训练完成后返回。
- `model_name_or_path`
  - 含义：实际使用的模型名称或权重路径。
  - 取值：字符串。
  - 出现时机：训练完成后返回。
- `hyperparams`
  - 含义：训练超参数对象。
  - 取值：对象。
  - 出现时机：训练完成后返回。
- `epochs_total`
  - 含义：总训练轮数。
  - 取值：整数或 `null`。
  - 出现时机：训练完成后返回。
- `epochs_done`
  - 含义：完成的轮数。
  - 取值：整数。
  - 出现时机：训练完成后返回。
- `best`
  - 含义：最佳指标对象。
  - 取值：对象或 `null`。
  - 出现时机：训练完成后返回。
- `artifacts`
  - 含义：产物文件列表。
  - 取值：数组，可能包含 `best.pt` / `last.pt` / `results.csv` / `args.yaml`。
  - 出现时机：训练完成后返回。
- `bundle.zip_path`
  - 含义：交付包路径，包含 `artifacts` 与 `meta.json`。
  - 取值：路径字符串。
  - 出现时机：训练完成后返回。
- `created_at`
  - 含义：训练开始时间（UTC）。
  - 取值：ISO 8601 字符串。
  - 出现时机：训练完成后返回。
- `finished_at`
  - 含义：训练结束时间（UTC）。
  - 取值：ISO 8601 字符串。
  - 出现时机：训练完成后返回。

**hyperparams 内逐字段注解**

- `epochs`：训练轮数。
- `imgsz`：训练图片尺寸。
- `batch`：batch 大小（已解析为数值）。
- `patience`：早停耐心值。
- `augment_level`：数据增强等级。
- `lr_scale`：学习率缩放倍数。
- `freeze`：冻结层数（continue 时可能出现）。
- `device_policy`：设备策略。
- `seed`：随机种子。
- `val`：是否执行验证。
- `save`：是否保存权重。
- `lr0`：实际学习率基值（根据 `lr_scale` 计算）。
- `workers`：数据加载 worker 数。

**best 内逐字段注解**

- `epoch`：最佳指标对应的 epoch。
- `map50`：mAP@0.5。
- `map50_95`：mAP@0.5:0.95。
- `precision`：精度。
- `recall`：召回率。

**artifacts 内文件说明**

- `best.pt`：最佳权重。
- `last.pt`：最后一轮权重。
- `results.csv`：训练指标记录。
- `args.yaml`：训练参数记录。

**典型调用流程中的位置**：progress 显示完成后，调用 result 获取产物与交付包路径。

**常见错误与排查**

1. `training not finished`：需等待任务完成。
2. `meta.json not found` / `meta.json invalid`：训练产出异常或文件损坏，建议重新训练。

---

## POST /infer/stream

**功能说明**：启动一个实时推理流，从RTSP URL读取视频流并使用指定模型进行推理。

### Request

- **Content-Type**：`application/json`

**请求体 JSON 示例**

```json
{
  "rtsp_url": "rtsp://example.com:554/stream1",
  "sample_fps": 2.0,
  "scenario": {
    "scenario_id": "traffic_monitoring",
    "models": [
      {
        "alias": "vehicle_detector",
        "model_id": "yolov8s_vehicles",
        "weights_path": "/path/to/weights/vehicle_detector.pt",
        "labels": ["car", "truck", "bus"],
        "params": {
          "conf": 0.3,
          "iou": 0.5,
          "imgsz": 640,
          "max_det": 100
        }
      },
      {
        "alias": "person_detector",
        "model_id": "yolov8s_persons",
        "weights_path": "/path/to/weights/person_detector.pt",
        "labels": ["person"],
        "params": {
          "conf": 0.25,
          "iou": 0.45,
          "imgsz": 640,
          "max_det": 50
        }
      }
    ]
  }
}
```

**逐字段注解**

| 字段 | 类型 | 必选 | 默认值 | 描述 |
|------|------|------|--------|------|
| `rtsp_url` | string | 是 | - | RTSP流URL地址 |
| `sample_fps` | number | 否 | 2.0 | 视频流采样帧率 |
| `scenario` | object | 是 | - | 推理场景配置 |
| `scenario.scenario_id` | string | 是 | - | 场景唯一标识符 |
| `scenario.models` | array | 是 | - | 场景中使用的模型列表 |
| `scenario.models[].alias` | string | 是 | - | 模型别名 |
| `scenario.models[].model_id` | string | 是 | - | 模型唯一标识符 |
| `scenario.models[].weights_path` | string | 是 | - | 模型权重文件路径 |
| `scenario.models[].labels` | array | 否 | null | 模型输出标签列表 |
| `scenario.models[].params` | object | 是 | - | 推理参数配置 |
| `scenario.models[].params.conf` | number | 否 | 0.25 | 置信度阈值 |
| `scenario.models[].params.iou` | number | 否 | 0.45 | IoU阈值 |
| `scenario.models[].params.imgsz` | integer | 否 | 640 | 推理图像大小 |
| `scenario.models[].params.max_det` | integer | 否 | 50 | 每帧最大检测数 |

### Response

**成功返回 JSON 示例**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running"
}
```

**逐字段注解**

| 字段 | 类型 | 描述 |
|------|------|------|
| `job_id` | string | 推理作业唯一标识符 |
| `status` | string | 作业启动状态，固定为 "running" |

**处理流程**

1. **参数验证**：检查 `rtsp_url` 是否提供
2. **作业创建**：调用 `job_manager.start_job()` 创建新作业
3. **模型构建**：调用 `_build_models()` 加载并配置指定的模型
4. **Webhook配置**：解析并验证 webhook URL
5. **线程启动**：
   - 创建并启动帧抓取线程 (`_frame_grabber_loop`)，从RTSP URL读取视频流
   - 创建并启动推理线程 (`_run_job`)，对抓取的帧进行模型推理
6. **结果返回**：返回作业ID和启动状态

**常见错误与排查**

| 错误状态码 | 错误信息 | 描述 |
|------------|----------|------|
| 400 | "rtsp_url is required" | 未提供 RTSP 流 URL |
| 400 | "Missing model config for alias '{alias}'" | 模型配置缺失 |
| 400 | "Missing params for model alias '{alias}'" | 模型参数缺失 |
| 400 | "Missing model_id for model alias '{alias}'" | 模型ID缺失 |
| 400 | "webhook_url must start with http:// or https://" | Webhook URL 格式不正确 |
| 400 | "webhook_url host cannot be 0.0.0.0" | Webhook URL 主机不能是 0.0.0.0 |

**注意事项**

1. 接口启动后会创建两个后台线程：
   - 帧抓取线程：持续从 RTSP URL 读取视频帧
   - 推理线程：对抓取的帧进行模型推理并发送结果

2. 推理结果会通过配置的 Webhook URL 发送回调

3. 可以通过 `/infer/{job_id}/stop` 接口停止正在运行的推理作业

4. 采样帧率 (`sample_fps`) 应设置为正数，建议根据硬件性能和网络带宽调整

5. 模型权重文件路径 (`weights_path`) 必须是系统可访问的有效路径

---

## POST /infer/{job_id}/stop

**功能说明**：停止指定的推理作业。

### Request

- **Content-Type**：无（POST 请求）

### Response

**成功返回 JSON 示例**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "stopped",
  "stopped_at": "2024-05-10T12:45:12Z"
}
```

**逐字段注解**

| 字段 | 类型 | 描述 |
|------|------|------|
| `job_id` | string | 推理作业唯一标识符 |
| `status` | string | 作业停止状态，固定为 "stopped" |
| `stopped_at` | string | 作业停止时间（UTC），ISO 8601 格式 |

**处理流程**

1. **参数验证**：检查 `job_id` 是否存在
2. **作业停止**：调用 `job_manager.stop_job()` 停止指定作业
3. **结果返回**：返回作业 ID、停止状态和停止时间

**常见错误与排查**

| 错误状态码 | 错误信息 | 描述 |
|------------|----------|------|
| 404 | "job not found" | 作业 ID 不存在 |

**典型调用流程中的位置**：当需要停止正在运行的推理作业时调用此接口。

---

## GET /preview/{job_id}

**功能说明**：获取推理作业的实时视频预览流，包含检测结果的可视化，复制接口使用浏览器打开，查看推理视频。

### Request

- **Content-Type**：无（GET 请求）

### Response

**成功返回**：
- **Content-Type**：`multipart/x-mixed-replace; boundary=frame`
- **响应体**：流式视频帧，每一帧为 JPEG 格式，包含检测结果的可视化叠加

**处理流程**

1. **参数验证**：检查 `job_id` 是否存在
2. **作业获取**：调用 `job_manager.get_job()` 获取作业实例
3. **流生成**：创建流式响应生成器，持续获取最新帧并添加检测结果可视化
4. **帧处理**：
   - 获取最新原始帧
   - 获取最新检测结果
   - 绘制检测结果到帧上
   - 调整帧大小（宽度超过 960 时缩放）
   - 编码为 JPEG 格式
   - 作为流式响应返回
5. **流控制**：当作业停止时，停止流传输

**常见错误与排查**

| 错误状态码 | 错误信息 | 描述 |
|------------|----------|------|
| 404 | "job not found" | 作业 ID 不存在 |

**典型调用流程中的位置**：启动推理作业后，通过此接口查看实时推理结果的可视化预览。

**注意事项**

1. 此接口返回的是流式响应，需要客户端支持 `multipart/x-mixed-replace` 格式
2. 视频流会在作业停止时自动终止
3. 为了优化传输性能，帧宽度超过 960 时会自动缩放
