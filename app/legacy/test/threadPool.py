from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from typing import Dict, Callable, Any
from pydantic import BaseModel

app = FastAPI()


# 自定义方法示例
def my_custom_task(stop_event: threading.Event, file_path: str):
    """自定义任务逻辑"""
    while not stop_event.is_set():
        # 这里替换为你的实际业务逻辑
        print(f"Processing with args: {file_path}, kwargs: {file_path}")
        # 模拟工作
        stop_event.wait(2)  
    print("Task completed gracefully")

class ThreadManager:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers)
        self.tasks: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def start_custom_task(
        self,
        task_func: Callable,  # 自定义方法
        *task_args,          # 方法的位置参数
        **task_kwargs         # 方法的关键字参数
    ) -> str:
        """启动自定义任务"""
        task_id = str(uuid.uuid4())
        stop_event = threading.Event()
        
        # 包装任务以添加生命周期管理
        def wrapped_task():
            try:
                task_func(stop_event, *task_args, **task_kwargs)
            except Exception as e:
                with self.lock:
                    self.tasks[task_id]["status"] = f"error: {str(e)}"
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
                "function": task_func.__name__
            }
        
        return task_id

    def stop_task(self, task_id: str) -> bool:
        """停止指定任务"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            self.tasks[task_id]["stop_event"].set()
            self.tasks[task_id]["status"] = "stopping"
            return True

    def get_status(self, task_id: str) -> dict:
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)
            return task if task else None

# 初始化管理器
task_manager = ThreadManager()

# API模型
class TaskRequest(BaseModel):
    args: list = []
    kwargs: dict = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    function: str

# API端点
@app.post("/start_custom_task", response_model=TaskResponse)
async def start_task(request: TaskRequest):
    task_id = task_manager.start_custom_task(
        my_custom_task,  # 传入自定义方法
        "1.txt"
    )
    task = task_manager.get_status(task_id)
    return {
        "task_id": task_id,
        "status": task["status"],
        "function": task["function"]
    }

@app.post("/stop_task/{task_id}")
async def stop_task(task_id: str):
    if not task_manager.stop_task(task_id):
        raise HTTPException(404, "Task not found")
    return {"status": "stopping"}

@app.get("/task_status/{task_id}", response_model=TaskResponse)
async def get_status(task_id: str):
    task = task_manager.get_status(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "task_id": task_id,
        "status": task["status"],
        "function": task["function"]
    }

if __name__ == "__main__":
    # main()
    #https://blog.csdn.net/jacksoon/article/details/142639053
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", reload=False, port=5001)  #,workers=20,log_level="info"