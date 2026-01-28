# T2.0R 本地数据集注册验收（register_local）

> 目标：覆盖 **register_local** 请求体字段说明、Windows 路径注意事项、resolved_data.yaml 生成规则与失败用例。

## A. 请求体字段说明

`POST /datasets/register_local`

```json
{
  "name": "smoking_v3",
  "root_dir": "D:/datasets/smoking.v3i.yolov8",
  "data_yaml": "data.yaml"
}
```

- `name`（可选）：数据集名称，仅记录在元数据中。
- `root_dir`（必填）：本地 YOLO 数据集根目录（包含 `train/`, `valid/` 或 `val/`, `test/`，以及 `data.yaml`）。
- `data_yaml`（可选）：
  - 为空时默认 `root_dir/data.yaml`。
  - 可以使用相对路径（相对 `root_dir`）或绝对路径。

### Windows 路径 JSON 转义

- 推荐使用 `/`：`D:/datasets/smoking.v3i.yolov8`
- 或使用双反斜杠：`D:\\datasets\\smoking.v3i.yolov8`
- **不要**使用单个 `\`，避免 `JSON Invalid \escape`。

---

## B. resolved_data.yaml 生成规则

- 生成位置：`datasets/<dataset_id>/resolved_data.yaml`
- 路径规则：
  - `train` 强制指向 `root_dir/train/images`
  - `val` 强制指向 `root_dir/valid/images`，若 `valid/` 不存在则使用 `root_dir/val/images`
  - `test` 仅在 `root_dir/test/images` 存在时写入
- 所有路径必须为绝对路径，且 **不允许出现 `../`**。
- `names` 与 `nc` 必须一致，否则拒绝注册。

---

## C. 失败用例验收

### C-1 root_dir 不存在

- 请求：`root_dir` 指向不存在路径
- 期望：HTTP 400，`root_dir invalid`

### C-2 data.yaml 不存在/找不到

- 请求：`data_yaml` 指向不存在文件
- 期望：HTTP 400，`missing data.yaml`

### C-3 train 结构缺失

- 缺少 `train/images` 或 `train/labels`
- 期望：HTTP 400，`train structure invalid`

### C-4 valid/val 缺失

- `valid/images` + `valid/labels` 不存在，且 `val/images` + `val/labels` 也不存在
- 期望：HTTP 400，`missing val/valid`

### C-5 nc 与 names 长度不一致

- `data.yaml` 中 `nc` 与 `names` 长度不一致
- 期望：HTTP 400，`nc names mismatch`
