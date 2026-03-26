import socketio
import logging

logger = logging.getLogger("model_forge.socket_manager")

class SocketManager:
    def __init__(self):
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins="*"
        )
        # 挂载路径设为空，由 FastAPI 的 mount 控制
        self.app = socketio.ASGIApp(self.sio, socketio_path='')
        # 用于存储 FastAPI 的主 loop
        self.loop = None

    # 确认这个方法名是 emit_event，且在类内部
    async def emit_event(self, event_name: str, data: dict, room: str = None):
        """
        核心发送方法，供异步调用
        """
        try:
            # 测试数据打印
            # print(f"📡 [SocketIO] 正在发送事件: {event_name} 到房间: {room}")
            await self.sio.emit(event_name, data, room=room)
        except Exception as e:
            logger.error(f"SocketIO 内部推送失败: {e}")

    def mount_to_fastapi(self, fastapi_app):
        # 挂载到 /ws/socket.io
        fastapi_app.mount("/ws/socket.io", self.app)

# 创建全局唯一实例
socket_manager = SocketManager()