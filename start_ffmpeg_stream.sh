#!/bin/bash

# 等待网络和 MediaMTX 服务启动
sleep 5

nohup ffmpeg -re -stream_loop -1 -i "/aidata/deduction/data/videos/device_1748309540692.mp4" -c copy -f rtsp rtsp://localhost:8554/craft3


: <<'EOF'
sudo vim /etc/systemd/system/MakeRtspDemo.service
[Unit]
Description=FFmpeg RTSP Streaming Service
After=network.target mediamtx.service
Requires=mediamtx.service

[Service]
Type=simple
User=lenovo
ExecStart=/usr/local/bin/start_rtsp_stream.sh
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=ffmpeg-rtsp

# 防止服务启动过快
StartLimitInterval=30
StartLimitBurst=5

[Install]
WantedBy=multi-user.target

# 重载 systemd 配置
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable MakeRtspDemo.service

# 立即启动服务
sudo systemctl start MakeRtspDemo.service

# 检查服务状态
systemctl status MakeRtspDemo.service
EOF


