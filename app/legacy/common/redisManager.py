import json

import redis
import logging
from typing import Any, Generator, Optional, Dict, List, Union
from contextlib import contextmanager
from .config import *

class RedisManager:
    """
    Redis 连接管理封装类（线程安全）
    功能：
    - 自动管理连接池
    - 支持重连机制
    - 常用操作封装
    - 上下文管理器支持
    - 异常处理与日志
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB_INDEX,
        password: Optional[str] = REDIS_PASSWORD,
        max_connections: int = REDIS_MAX_CONN,
        decode_responses: bool = True
    ):
        """
        初始化 Redis 连接池
        :param max_connections: 连接池最大连接数
        """
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=decode_responses
        )
        self.logger = logging.getLogger('RedisManager')
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    @contextmanager
    def get_connection(self) -> Generator[redis.Redis, None, None]:
        """
        获取 Redis 连接的上下文管理器
        用法：
        with redis_manager.get_connection() as conn:
            conn.set('key', 'value')
        """
        conn = None
        try:
            conn = redis.Redis(connection_pool=self._pool)
            yield conn
        except redis.RedisError as e:
            self.logger.error(f"Redis 操作失败: {str(e)}")
            raise
        finally:
            if conn:
                # 实际连接由连接池管理，不需要手动关闭
                pass

    def retry_on_failure(retries=3):
        """
        重试装饰器，用于关键操作
        """
        def decorator(func):
            def wrapper(self,*args, **kwargs):
                last_exception = None
                for _ in range(retries):
                    try:
                        return func(self,*args, **kwargs)
                    except (redis.ConnectionError, redis.TimeoutError) as e:
                        last_exception = e
                        self.logger.warning(f"连接异常，尝试重试... ({_+1}/{retries})")
                        continue
                raise last_exception or RuntimeError("未知错误")
            return wrapper
        return decorator

    # 常用操作封装
    @retry_on_failure()
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        with self.get_connection() as conn:
            return conn.set(key, json.dumps(value, ensure_ascii=False), ex=ex)

    @retry_on_failure()
    def get(self, key: str) -> Optional[str]:
        with self.get_connection() as conn:
            data = conn.get(key)
            return json.loads(data) if data else None

    # @retry_on_failure()
    # def get(self, key: str) -> str:
    #     data = self.getOrginalValue(key)
    #     return json.loads(data) if data else None

    @retry_on_failure()
    def hmset(self, name: str, mapping: Dict[str, Any]) -> bool:
        with self.get_connection() as conn:
            return conn.hmset(name, mapping)

    @retry_on_failure()
    def hgetall(self, name: str) -> Dict[str, str]:
        with self.get_connection() as conn:
            return conn.hgetall(name)


    @retry_on_failure()
    def delete(self, *keys: str) -> int:
        with self.get_connection() as conn:
            return conn.delete(*keys)


    @retry_on_failure()
    def exists(self, key: str) -> int:
        with self.get_connection() as conn:
            return conn.exists(key)


    @retry_on_failure()
    def pipeline(self) -> redis.client.Pipeline:
        with self.get_connection() as conn:
            return conn.pipeline()

    # 高级功能
    def atomic_transaction(self, key: str, callback: callable):
        """
        执行原子事务
        :param key: 需要监视的键
        :param callback: 事务处理函数，接收 pipeline 参数
        """
        with self.get_connection() as conn:
            while True:
                try:
                    conn.watch(key)
                    pipeline = conn.pipeline()
                    callback(pipeline)
                    pipeline.execute()
                    break
                except redis.WatchError:
                    continue

    # 统计信息
    @property
    def connection_info(self) -> Dict[str, Union[int, List[Dict]]]:
        """
        获取连接池统计信息
        """
        return {
            'max_connections': self._pool.max_connections,
            'current_connections': self._pool._created_connections,
            'available': len(self._pool._available_connections),
            'in_use': self._pool._in_use_connections
        }