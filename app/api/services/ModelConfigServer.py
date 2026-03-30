from app.api.services.dbHelp import db


# 生成模型表（独立表，无外键关联）
MODEL_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键，自增ID',
    alias VARCHAR(64) NOT NULL COMMENT '模型别名（如helmet、person等）',
    model_id VARCHAR(64) NOT NULL COMMENT '模型唯一标识',
    weights_path VARCHAR(512) NOT NULL COMMENT '模型权重路径',
    labels TEXT COMMENT '类别标签（JSON数组）',
    params TEXT COMMENT '模型参数（JSON对象，conf、iou、imgsz等）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型配置表';
"""

class ModelConfigDB:


    _fileds =" id, alias, model_id, weights_path, labels, params, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at "
    """
    模型配置表(models)的操作类
    """
    def create_model(self, alias, model_id, weights_path, labels=None, params=None):
        sql = """
            INSERT INTO models (alias, model_id, weights_path, labels, params)
            VALUES (%s, %s, %s, %s, %s)
        """
        db.execute(sql, (alias, model_id, weights_path, labels, params))
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_model(self, model_id):
        sql = f"SELECT { self._fileds } FROM models WHERE model_id=%s"
        return db.fetch_dict(sql, (model_id,))

    def list_models(self):
        sql = f"SELECT { self._fileds } FROM models ORDER BY id DESC"
        return db.fetch_dict(sql)

    def update_model(self, model_id, alias=None, weights_path=None, labels=None, params=None):
        fields = []
        p = []
        if alias is not None:
            fields.append('alias=%s')
            p.append(alias)
        if weights_path is not None:
            fields.append('weights_path=%s')
            p.append(weights_path)
        if labels is not None:
            fields.append('labels=%s')
            p.append(labels)
        if params is not None:
            fields.append('params=%s')
            p.append(params)
        if not fields:
            return 0
        sql = f"UPDATE models SET {', '.join(fields)} WHERE model_id=%s"
        p.append(model_id)
        return db.execute(sql, tuple(p))

    def delete_model(self, model_id):
        sql = "DELETE FROM models WHERE model_id=%s"
        return db.execute(sql, (model_id,))


model_config_db = ModelConfigDB()