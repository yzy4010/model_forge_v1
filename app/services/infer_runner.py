import time
from threading import Lock
from typing import Any, Dict, Optional

from app.services.video_stream_legacy.common.redisManager import RedisManager
from app.services.video_stream_legacy.common.threadTaskManager import ThreadTaskManager
from app.services.video_stream_legacy.common.config import REDIS_KEY_PROCESS_STATUS
from app.services.video_stream_legacy.java_config_client import (
    fetch_all_processor_configs_from_java,
    fetch_processor_config_from_java,
)

# 通过 sys.path 兼容后，可以直接这样导入 legacy 推理入口
from process_videoStream import start_process_video_stream  # type: ignore[import]

# 启动后诊断：若出现异常写入 _task_errors，则在 /infer/start 中返回 error；
# 任务连续 running 超过 _START_STABLE_SEC 即认为启动成功，避免成功路径长时间阻塞。
_START_DIAG_WAIT_SEC = 3.0
_START_DIAG_POLL_SEC = 0.05
_START_STABLE_SEC = 0.2


class InferRunner:
    """
    手动启停的视频流推理编排器：
    - start(cameras_id): 从 Java 拉配置 -> 启动一条推理线程
    - stop(cameras_id):  写 Redis statusKey='end' + 通知线程池停止
    - status(): 返回当前所有任务状态
    """

    def __init__(self, redis: RedisManager):
        self.redis = redis
        self.tm = ThreadTaskManager(max_workers=20, thread_name_prefix="manualInfer")
        self._lock = Lock()
        # camerasId -> taskid
        self._tasks: Dict[str, str] = {}
        # camerasId -> 最近一次推理线程异常说明（供 start 短时检测与 status 展示）
        self._task_errors: Dict[str, str] = {}

    # 启动一条推理
    def start(self, cameras_id: str) -> dict:
        with self._lock:
            # 已在运行则直接返回
            old_taskid = self._tasks.get(cameras_id)
            if old_taskid is not None:
                info = self.tm.get_task_info(old_taskid)
                if info is not None and info.get("status") in ("running", "pending"):
                    return {
                        "ok": True,
                        "msg": f"摄像头 {cameras_id} 已在运行",
                        "taskid": old_taskid,
                        "error": None,
                    }

            cfg = fetch_processor_config_from_java(cameras_id)
            if cfg is None:
                return {
                    "ok": False,
                    "msg": f"Java 未返回 camerasId={cameras_id} 的配置",
                    "error": None,
                    "taskid": None,
                }

            self._task_errors.pop(cameras_id, None)

            # 在 Python 侧为 cfg.id 赋值，用于 Redis key/statusKey
            try:
                cfg.id = int(cameras_id)
            except Exception:
                cfg.id = 0

            taskid = f"infer_{cameras_id}"

            # 清理残留的停止信号，避免一启动就退出
            status_key = REDIS_KEY_PROCESS_STATUS + str(cfg.id)
            if self.redis.get(status_key) == "end":
                self.redis.delete(status_key)

            new_taskid = self.tm.start_task(self._run_one, 0, taskid, cameras_id, cfg)
            self._tasks[cameras_id] = new_taskid

        err = self._wait_start_diagnostic(cameras_id, new_taskid)
        if err is not None:
            with self._lock:
                self._tasks.pop(cameras_id, None)
                # 保留 _task_errors，便于 GET /status 的 task_errors 对照；下次 start 会清空
            return {
                "ok": False,
                "msg": "推理任务异常退出",
                "taskid": new_taskid,
                "error": err,
            }

        return {"ok": True, "msg": "启动成功", "taskid": new_taskid, "error": None}

    # 停止一条推理
    def stop(self, cameras_id: str, wait_seconds: int = 15) -> dict:
        with self._lock:
            taskid = self._tasks.get(cameras_id)
            if not taskid:
                return {"ok": True, "msg": f"摄像头 {cameras_id} 无运行任务"}

            # 计算对应的 cfg.id（和 start 里保持一致）
            try:
                cfg_id = int(cameras_id)
            except Exception:
                cfg_id = 0

            status_key = REDIS_KEY_PROCESS_STATUS + str(cfg_id)

            # 1) 发送停止信号（legacy 循环会轮询这个 key）
            self.redis.set(status_key, "end")

            # 2) 通知线程池停止（给 stop_event 用）
            self.tm.stop_task(taskid)

        # 3) 等待任务真正退出
        end_at = time.time() + wait_seconds
        while time.time() < end_at:
            info = self.tm.get_task_info(taskid)
            if info is None:
                return {"ok": True, "msg": "已停止"}

            status = info.get("status")
            if status in ("stopped", "finished", "error"):
                return {"ok": True, "msg": f"已停止（状态={status}）"}

            time.sleep(0.3)

        return {"ok": False, "msg": "停止超时（已发送 end 信号，线程可能仍在退出中）"}

    # 查询所有任务状态
    def status(self) -> dict:
        tasks = self.tm.list_all_tasks()
        with self._lock:
            errors = dict(self._task_errors)
        return {"ok": True, "tasks": tasks, "task_errors": errors}

    def _wait_start_diagnostic(self, cameras_id: str, taskid: str) -> Optional[str]:
        """在启动后短时间内检测推理线程是否已因异常退出。"""
        deadline = time.time() + _START_DIAG_WAIT_SEC
        running_since: Optional[float] = None
        while time.time() < deadline:
            with self._lock:
                err = self._task_errors.get(cameras_id)
            if err is not None:
                return err
            info = self.tm.get_task_info(taskid)
            if info is not None:
                st = info.get("status")
                if st == "failed":
                    with self._lock:
                        err = self._task_errors.get(cameras_id)
                    return err or (info.get("error") or "unknown")
                if st in ("running", "stopping"):
                    now = time.time()
                    if running_since is None:
                        running_since = now
                    elif now - running_since >= _START_STABLE_SEC:
                        return None
            else:
                with self._lock:
                    err = self._task_errors.get(cameras_id)
                if err is not None:
                    return err
                # 已从调度表移除且无异常记录：视为快速正常结束（如极短视频）
                return None
            time.sleep(_START_DIAG_POLL_SEC)
        return None

    # 从 Java 拉取全部摄像头推理配置（与手动启停共用同一数据源）
    def list_all_java_processor_configs(self) -> dict:
        items = fetch_all_processor_configs_from_java()
        if items is None:
            return {"ok": False, "msg": "Java 未返回有效配置列表", "count": 0, "items": []}
        return {
            "ok": True,
            "count": len(items),
            "items": [c.model_dump(mode="json") for c in items],
        }

    # 线程池里的真实执行函数
    def _run_one(self, stop_event, cameras_id: str, cfg: Any):
        """
        退出条件：
        - legacy 内部读到 Redis statusKey='end' 主动 break
        - 或发生异常（写入 _task_errors 后抛出，供 /infer/start 短时诊断）
        """
        try:
            start_process_video_stream(cfg, self.redis)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[InferRunner] 推理任务异常退出: {err!r}")
            with self._lock:
                self._task_errors[cameras_id] = err
            raise
