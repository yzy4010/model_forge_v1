from typing import Any, Dict, List, Optional, Tuple
import asyncio
import tracemalloc
import json

from fastapi import WebSocket

class ConnectionManager:
    """
    支持：
    - 客户端连接的唯一映射（支持标识符，比如user_id/session_id）
    - 分组管理（群组消息）
    - 点对点消息发送
    """

    def __init__(self):
        # 连接: {client_id: websocket}
        self.connections: Dict[str, WebSocket] = {}
        # 分组: {group: set(client_id)}
        self.groups: Dict[str, set] = {}

        self.lock = asyncio.Lock()  # 保护多协程环境的并发安全

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None, group: Optional[str] = None):
        """
        建立连接，分配唯一client_id，并可选地加入某个group
        """
        await websocket.accept()
        # client_id 必须唯一（比如登录的用户ID、session_id、或随机生成）
        if client_id is None:
            client_id = id(websocket).__str__()
        async with self.lock:
            self.connections[client_id] = websocket
            if group:
                if group not in self.groups:
                    self.groups[group] = set()
                self.groups[group].add(client_id)
        return client_id

    async def disconnect(self, client_id: str):
        """
        断开客户端（释放连接并移除所有分组引用）
        """
        async with self.lock:
            ws = self.connections.pop(client_id, None)
            for group_name in list(self.groups.keys()):
                self.groups[group_name].discard(client_id)
                if not self.groups[group_name]:
                    del self.groups[group_name]
            # 关闭WebSocket
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass

    async def remove_connection(self, websocket: WebSocket):
        """
        根据WebSocket移除，兼容旧接口（内部转换为client_id删除）
        """
        async with self.lock:
            found_id = None
            for k, v in self.connections.items():
                if v is websocket:
                    found_id = k
                    break
            if found_id:
                await self.disconnect(found_id)

    async def add_to_group(self, client_id: str, group: str):
        async with self.lock:
            if group not in self.groups:
                self.groups[group] = set()
            self.groups[group].add(client_id)

    async def remove_from_group(self, client_id: str, group: str):
        async with self.lock:
            if group in self.groups:
                self.groups[group].discard(client_id)
                if not self.groups[group]:
                    del self.groups[group]

    async def broadcast(self, message: str, group: Optional[str] = None):
        """
        向所有活跃连接 或 指定分组 广播消息
        """
        async with self.lock:
            if group:
                ids = self.groups.get(group, set()).copy()
            else:
                ids = set(self.connections.keys())
            connections = [self.connections[cid] for cid in ids if cid in self.connections]
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                # 这里可触发断线回收
                pass

    async def broadcast_json(self, message: str, group: Optional[str] = None):
        """
        向所有活跃连接/分组 广播JSON消息
        """
        async with self.lock:
            if group:
                ids = self.groups.get(group, set()).copy()
            else:
                ids = set(self.connections.keys())
            connections = [self.connections[cid] for cid in ids if cid in self.connections]
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def send_to(self, client_id: str, message: str):
        """
        点对点消息发送
        """
        async with self.lock:
            ws = self.connections.get(client_id)
        if ws:
            try:
                await ws.send_text(message)
            except Exception:
                pass

    async def send_json_to(self, client_id: str, message: Any):
        """
        点对点发送JSON消息
        """
        async with self.lock:
            ws = self.connections.get(client_id)
        if ws:
            try:
                # message应已为str或dict
                if isinstance(message, (dict, list)):
                    await ws.send_text(json.dumps(message, ensure_ascii=False))
                else:
                    await ws.send_text(str(message))
            except Exception:
                pass

manager = ConnectionManager()



