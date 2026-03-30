

from app.api.services.dbHelp import db

SCENE_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scene_configs (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键，自增ID',
    scene_id VARCHAR(64) NOT NULL UNIQUE COMMENT '场景唯一ID（如scenario_id）',
    rtsp_url VARCHAR(512) NOT NULL COMMENT 'RTSP视频流地址',
    sample_fps INT DEFAULT 4 COMMENT '抽帧频率',
    scenario_name VARCHAR(128) COMMENT '场景名称，可选',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推理场景主配置表';
"""


class SceneConfigDB:

    _fileds =""" id, scene_id,rtsp_url, sample_fps, scenario_name, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at """

    """
    推理场景主配置表(scene_configs)相关操作方法
    """
    def create_scene_config(self, scene_id, rtsp_url, sample_fps=4, scenario_name=None):
        sql = """
            INSERT INTO scene_configs (scene_id,rtsp_url, sample_fps, scenario_name)
            VALUES (%s, %s, %s, %s)
        """
        params = (scene_id, rtsp_url, sample_fps, scenario_name)
        db.execute(sql, params)
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_scene_config(self, scene_id):
        sql = f"SELECT { self._fileds } FROM scene_configs WHERE scene_id=%s"
        return db.fetch_dict(sql, (scene_id,))

    def update_scene_config(self, scene_id, rtsp_url=None, sample_fps=None, scenario_name=None):
        fields = []
        params = []
        if rtsp_url is not None:
            fields.append('rtsp_url=%s')
            params.append(rtsp_url)
        if sample_fps is not None:
            fields.append('sample_fps=%s')
            params.append(sample_fps)
        if scenario_name is not None:
            fields.append('scenario_name=%s')
            params.append(scenario_name)
        if not fields:
            return 0
        sql = f"UPDATE scene_configs SET {', '.join(fields)} WHERE scene_id=%s"
        params.append(scene_id)
        return db.execute(sql, tuple(params))

    def delete_scene_config(self, scene_id):
        sql = "DELETE FROM scene_configs WHERE scene_id=%s"
        return db.execute(sql, (scene_id,))

    def list_scene_configs(self):
        sql = f"SELECT { self._fileds } FROM scene_configs ORDER BY id DESC"
        return db.fetch_dict(sql)


scene_config_db = SceneConfigDB()