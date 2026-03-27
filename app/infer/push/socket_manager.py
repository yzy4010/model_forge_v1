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

        # 注册基础事件（连接/订阅/退订）
        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.sio.event
        async def connect(sid, environ, auth):  # type: ignore[no-redef]
            logger.info("SocketIO connect sid=%s auth=%s", sid, bool(auth))

        @self.sio.event
        async def disconnect(sid):  # type: ignore[no-redef]
            logger.info("SocketIO disconnect sid=%s", sid)

        @self.sio.event
        async def subscribe(sid, data):  # type: ignore[no-redef]
            """
            前端调用：socket.emit("subscribe", {"job_id": "<job_id>"})
            服务端将该连接加入 job_id 房间，从而接收 room=job_id 的推送。
            """
            try:
                job_id = ""
                if isinstance(data, dict):
                    job_id = str(data.get("job_id") or "").strip()
                else:
                    job_id = str(data or "").strip()

                if not job_id:
                    await self.sio.emit(
                        "subscribe_ack",
                        {"ok": False, "error": "job_id is required"},
                        to=sid,
                    )
                    return

                await self.sio.enter_room(sid, job_id)
                logger.info("SocketIO subscribed sid=%s room=%s", sid, job_id)
                await self.sio.emit("subscribe_ack", {"ok": True, "job_id": job_id}, to=sid)
            except Exception as e:
                logger.exception("SocketIO subscribe failed sid=%s err=%s", sid, e)
                await self.sio.emit(
                    "subscribe_ack",
                    {"ok": False, "error": "subscribe failed"},
                    to=sid,
                )

        @self.sio.event
        async def unsubscribe(sid, data):  # type: ignore[no-redef]
            """
            前端调用：socket.emit("unsubscribe", {"job_id": "<job_id>"})
            """
            try:
                job_id = ""
                if isinstance(data, dict):
                    job_id = str(data.get("job_id") or "").strip()
                else:
                    job_id = str(data or "").strip()

                if not job_id:
                    await self.sio.emit(
                        "unsubscribe_ack",
                        {"ok": False, "error": "job_id is required"},
                        to=sid,
                    )
                    return

                await self.sio.leave_room(sid, job_id)
                logger.info("SocketIO unsubscribed sid=%s room=%s", sid, job_id)
                await self.sio.emit("unsubscribe_ack", {"ok": True, "job_id": job_id}, to=sid)
            except Exception as e:
                logger.exception("SocketIO unsubscribe failed sid=%s err=%s", sid, e)
                await self.sio.emit(
                    "unsubscribe_ack",
                    {"ok": False, "error": "unsubscribe failed"},
                    to=sid,
                )

    # 确认这个方法名是 emit_event，且在类内部
    async def emit_event(self, event_name: str, data: dict, room: str = None):
        """
        核心发送方法，供异步调用
        """
        try:
            logger.debug("SocketIO emit event=%s room=%s", event_name, room)
            await self.sio.emit(event_name, data, room=room)
        except Exception as e:
            logger.error(f"SocketIO 内部推送失败: {e}")

    def mount_to_fastapi(self, fastapi_app):
        # 挂载到 /ws/socket.io
        fastapi_app.mount("/ws/socket.io", self.app)

# 创建全局唯一实例
socket_manager = SocketManager()