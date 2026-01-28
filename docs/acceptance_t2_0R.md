# T2.0R 端到端验收清单（本地路径注册）

> 目标：覆盖本地路径注册、失败场景以及新训/继续训端到端流程。

## A. 注册本地数据集（成功）

调用 `POST /datasets/register_local`：

- `root_dir` 指向一个真实 YOLO 数据集目录（包含 train/valid/test + data.yaml）

**期望**

- 返回 `dataset_id`
- 磁盘存在：
  - `datasets/<dataset_id>/dataset_meta.json`
  - `datasets/<dataset_id>/resolved_data.yaml`
- `resolved_data.yaml`：
  - train/val/test 路径均指向 `root_dir` 下（绝对路径）
  - 不包含 `../`
- `GET /datasets/{dataset_id}` 能返回同样信息

## B. 注册失败场景

- `root_dir` 不存在 -> 400（提示 `root_dir invalid`）
- `data.yaml` 不存在 -> 400（提示 `missing data.yaml`）
- `train/images` 或 `train/labels` 不存在 -> 400（提示 `train structure invalid`）
- `valid`/`val` 缺失 -> 400（提示 `missing val/valid`）
- `nc` 与 `names` 不一致 -> 400

## C. 端到端训练（新训）

- 使用上一步 `dataset_id` 调用 `POST /train/yolo/new`，`epochs=2`
- `/train/{job_id}/progress`：
  - `epochs_done` 从 0 -> 1 -> 2
- `/train/{job_id}/result`（completed）：
  - `artifacts` 包含 `best.pt`/`last.pt`/`results.csv`/`args.yaml`（存在则列）
  - `bundle.zip_path` 指向存在的 zip 文件

## D. 端到端训练（继续训 finetune_best）

- 用 `base_job_id` 调用 `POST /train/yolo/continue`，`epochs=2`
- 返回新 `job_id`，且不覆盖 `base_job` 输出
- `result` 完整同上
