from fastapi import Depends
from ..common.redisManager import RedisManager

def get_redis() -> RedisManager:
    return RedisManager()

RedisDep = Depends(get_redis)