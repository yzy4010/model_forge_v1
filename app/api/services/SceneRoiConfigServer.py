from app.api.services.dbHelp import db


ROI_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scene_roi_configs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    scene_id VARCHAR(64) NOT NULL COMMENT '关联场景ID',
    camera_id VARCHAR(64) NOT NULL COMMENT '摄像头ID',
    config_version INT DEFAULT 1,
    resolution_width INT DEFAULT 640,
    resolution_height INT DEFAULT 360,
    roi_id VARCHAR(64) NOT NULL COMMENT '区域ID',
    name VARCHAR(128) NOT NULL COMMENT '区域名称',
    semantic_tag VARCHAR(64) COMMENT '语义标记',
    enabled BOOLEAN DEFAULT TRUE,
    geometry TEXT COMMENT '区域几何（如points点集，JSON数组）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='推理场景下的ROI区域配置表';
"""


class SceneRoiConfigDB:

    _fileds =""" id, scene_id, camera_id, config_version, resolution_width, resolution_height,
                roi_id, name, semantic_tag, enabled, geometry, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at """

    """
    推理场景下的ROI区域配置表(scene_roi_configs)相关操作方法
    """
    def create_roi(self, scene_id, camera_id, roi_id, name, semantic_tag=None, enabled=True, config_version=1,
                   resolution_width=640, resolution_height=360, geometry=None):
        sql = """
            INSERT INTO scene_roi_configs (
                scene_id, camera_id, config_version, resolution_width, resolution_height,
                roi_id, name, semantic_tag, enabled, geometry
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        db.execute(sql, (scene_id, camera_id, config_version, resolution_width, resolution_height,
                         roi_id, name, semantic_tag, enabled, geometry))
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_roi(self, roi_id):
        sql = f"SELECT { self._fileds } FROM scene_roi_configs WHERE roi_id=%s"
        return db.fetch_dict(sql, (roi_id,))

    def list_rois(self, scene_id=None):
        if scene_id:
            sql = f"SELECT { self._fileds } FROM scene_roi_configs WHERE scene_id=%s ORDER BY id DESC"
            return db.fetch_dict(sql, (scene_id,))
        else:
            sql = f"SELECT { self._fileds } FROM scene_roi_configs ORDER BY id DESC"
            return db.fetch_dict(sql)

    def update_roi(self, roi_id, name=None, semantic_tag=None, enabled=None, geometry=None):
        fields = []
        params = []
        if name is not None:
            fields.append('name=%s')
            params.append(name)
        if semantic_tag is not None:
            fields.append('semantic_tag=%s')
            params.append(semantic_tag)
        if enabled is not None:
            fields.append('enabled=%s')
            params.append(enabled)
        if geometry is not None:
            fields.append('geometry=%s')
            params.append(geometry)
        if not fields:
            return 0
        sql = f"UPDATE scene_roi_configs SET {', '.join(fields)} WHERE roi_id=%s"
        params.append(roi_id)
        return db.execute(sql, tuple(params))

    def delete_roi(self, roi_id):
        sql = "DELETE FROM scene_roi_configs WHERE roi_id=%s"
        return db.execute(sql, (roi_id,))


scene_roi_config_db = SceneRoiConfigDB()