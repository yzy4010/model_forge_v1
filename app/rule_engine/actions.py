from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def alert_action(rule_id, event_data, params):
    # 从 params 获取 JSON 中配置的自定义信息，如果没有则使用默认值
    # level = params.get("level", "info").upper()
    # message = params.get("message", "未定义的规则触发")

    level = params.get("level").upper()
    message = params.get("message")
    logger.debug(f"[{level} 告警] {message}")
    logger.debug(f"规则 ID: {rule_id}")
    logger.debug(f"告警内容: {message}")
    logger.debug(f"触发时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # print(f"--- [🚨 {level} 告警] ---")
    # print(f"规则 ID: {rule_id}")
    # print(f"告警内容: {message}")
    # print(f"触发时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 这里可以扩展：发送到钉钉、飞书、或存储到数据库


def save_image_action(rule_id, event_data, params):
    """保存图片动作"""
    # 此函数未使用
    # 功能已实现，在_run_job中直接执行保存逻辑，只保存触发规则的结果图片
    # 暂时只打日志，不执行耗时的 IO 操作
    logger.warning(f"规则 {rule_id} 触发了 save_image 动作，但功能尚未实现。")
    return None