from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Timer
from typing import Callable, Dict, Any
import uuid
import time


class ThreadTaskManager:
    def __init__(self, max_workers=4, thread_name_prefix="threadPool"):
        self.executor = ThreadPoolExecutor(max_workers, thread_name_prefix)
        self.tasks: Dict[str, dict] = {}
        self.lock = Lock()

    def start_task(
            self,
            task_func: Callable,  # 自定义方法
            timeout: int = 0,
            task_id: str = None,
            *task_args,  # 方法的位置参数
            **task_kwargs  # 方法的关键字参数
    ) -> str:
        """启动自定义任务"""
        if not task_id:
            task_id = uuid.uuid4().hex
        stop_event = Event()

        # 包装任务以添加生命周期管理
        def wrapped_task():
            try:
                task_func(stop_event, *task_args, **task_kwargs)
            except Exception as e:
                with self.lock:
                    self.tasks[task_id]["status"] = "failed"
                    self.tasks[task_id]["error"] = str(e)
            finally:
                with self.lock:
                    if task_id in self.tasks:
                        del self.tasks[task_id]

        future = self.executor.submit(wrapped_task)

        with self.lock:
            self.tasks[task_id] = {
                "future": future,
                "stop_event": stop_event,
                "status": "running",
                "function": task_func.__name__,
                "error": ""
            }
        if timeout > 0:
            timer = Timer(
                timeout,
                lambda: self.stop_task(task_id)
            )
            timer.start()
        return task_id

    def stop_task(self, task_id: str) -> bool:
        """停止指定任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            self.tasks[task_id]["stop_event"].set()
            self.tasks[task_id]["status"] = "stopping"
            self.tasks[task_id]["error"] = ""
            return True

    def get_task_info(self, task_id: str) -> dict:
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            return task if task else None

    def list_all_tasks(self) -> list:
        """列出所有任务（包含已完成）"""
        return list(self.tasks.items())

    def shutdown(self):
        """安全关闭线程池"""
        self.executor.shutdown(wait=True)
