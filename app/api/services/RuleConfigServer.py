
from app.api.services.dbHelp import db


# 生成规则配置表（独立表，无外键关联）
RULE_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rule_configs (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键，自增ID',
    rule_id VARCHAR(64) NOT NULL COMMENT '规则唯一ID',
    name VARCHAR(128) NOT NULL COMMENT '规则名称',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    conditions TEXT COMMENT '规则条件，JSON对象（如alias、roi等）',
    action_type VARCHAR(64) COMMENT '动作类型（如alert）',
    action_params TEXT COMMENT '动作参数JSON（如level, message, cooldown等）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规则配置表';
"""

class RuleConfigDB:
    _fileds =""" id,rule_id, name, enabled, conditions, action_type, action_params, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at """

    """
    规则配置表(rule_configs)的操作类
    """
    def create_rule(self, rule_id, name, enabled=True, conditions=None, action_type=None, action_params=None):
        sql = """
            INSERT INTO rule_configs (
                rule_id, name, enabled, conditions, action_type, action_params
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        db.execute(sql, (rule_id, name, enabled, conditions, action_type, action_params))
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_rule(self, rule_id):
        sql = f"SELECT { self._fileds } FROM rule_configs WHERE rule_id=%s"
        return db.fetch_dict(sql, (rule_id,))

    def list_rules(self):
        sql = f"SELECT { self._fileds } FROM rule_configs ORDER BY id DESC"
        return db.fetch_dict(sql)

    def update_rule(self, rule_id, name=None, enabled=None, conditions=None, action_type=None, action_params=None):
        fields = []
        params = []
        if name is not None:
            fields.append('name=%s')
            params.append(name)
        if enabled is not None:
            fields.append('enabled=%s')
            params.append(enabled)
        if conditions is not None:
            fields.append('conditions=%s')
            params.append(conditions)
        if action_type is not None:
            fields.append('action_type=%s')
            params.append(action_type)
        if action_params is not None:
            fields.append('action_params=%s')
            params.append(action_params)
        if not fields:
            return 0
        sql = f"UPDATE rule_configs SET {', '.join(fields)} WHERE rule_id=%s"
        params.append(rule_id)
        return db.execute(sql, tuple(params))

    def delete_rule(self, rule_id):
        sql = "DELETE FROM rule_configs WHERE rule_id=%s"
        return db.execute(sql, (rule_id,))


rule_config_db = RuleConfigDB()