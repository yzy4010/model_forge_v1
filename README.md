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
