"""全局配置入口。仅放跨模块的全局变量，各模块自有配置留在模块内部。"""
import os

# 部署类型
PUBLIC_TYPE = os.getenv("PUBLIC_TYPE", "2")  # 1=推理, 2=训练
