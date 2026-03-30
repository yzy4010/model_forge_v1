from app.api.services.dbHelp import db

MODEL_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scene_models (
    id INT PRIMARY KEY AUTO_INCREMENT,
    scene_id VARCHAR(64) NOT NULL COMMENT '关联场景ID',
    alias VARCHAR(64) NOT NULL COMMENT '模型别名（如helmet、person等）',
    model_id VARCHAR(64) NOT NULL COMMENT '模型唯一标识',
    weights_path VARCHAR(512) NOT NULL COMMENT '模型权重路径',
    labels TEXT COMMENT '类别标签（JSON数组）',
    params TEXT COMMENT '模型参数（JSON对象，conf、iou、imgsz等）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推理场景下的模型配置表';
"""




class SceneModelConfigDB:

    _fileds =""" id, scene_id, alias, model_id, weights_path, labels, params, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at """


    """
    推理场景下的模型配置表(scene_models)相关操作方法
    """
    def create_model(self, scene_id, alias, model_id, weights_path, labels=None, params=None):
        sql = """
            INSERT INTO scene_models (scene_id, alias, model_id, weights_path, labels, params)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        db.execute(sql, (scene_id, alias, model_id, weights_path, labels, params))
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_model(self, model_id):
        sql = f"SELECT { self._fileds } FROM scene_models WHERE model_id=%s"
        return db.fetch_dict(sql, (model_id,))

    def list_models(self, scene_id=None):
        if scene_id:
            sql = f"SELECT { self._fileds } FROM scene_models WHERE scene_id=%s ORDER BY id DESC"
            return db.fetch_dict(sql, (scene_id,))
        else:
            sql = f"SELECT { self._fileds } FROM scene_models ORDER BY id DESC"
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
        sql = f"UPDATE scene_models SET {', '.join(fields)} WHERE model_id=%s"
        p.append(model_id)
        return db.execute(sql, tuple(p))

    def delete_model(self, model_id):
        sql = "DELETE FROM scene_models WHERE model_id=%s"
        return db.execute(sql, (model_id,))

scene_model_config_db = SceneModelConfigDB()