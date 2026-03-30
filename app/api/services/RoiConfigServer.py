

from app.api.services.dbHelp import db



# 生成ROI区域配置表（独立表，无外键关联）
ROI_CONFIG_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS roi_configs (
    id INT PRIMARY KEY AUTO_INCREMENT COMMENT '主键，自增ID',
    camera_id VARCHAR(64) NOT NULL COMMENT '摄像头ID',
    config_version INT DEFAULT 1 COMMENT '配置版本号',
    resolution_width INT DEFAULT 640 COMMENT '分辨率宽度',
    resolution_height INT DEFAULT 360 COMMENT '分辨率高度',
    roi_id VARCHAR(64) NOT NULL COMMENT '区域ID',
    name VARCHAR(128) NOT NULL COMMENT '区域名称',
    semantic_tag VARCHAR(64) COMMENT '语义标记',
    enabled BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    geometry TEXT COMMENT '区域几何（如points点集，JSON数组）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ROI区域配置表';
"""

class RoiConfigDB:


    _fileds =""" id,camera_id, config_version, resolution_width, resolution_height, 
                roi_id, name, semantic_tag, enabled, geometry, DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i:%%s') AS created_at """

    """
    ROI区域配置表(roi_configs)的操作类
    """
    def create_roi(self, camera_id, roi_id, name, semantic_tag=None, enabled=True, config_version=1,
                   resolution_width=640, resolution_height=360, geometry=None):
        sql = """
            INSERT INTO roi_configs (
                camera_id, config_version, resolution_width, resolution_height, 
                roi_id, name, semantic_tag, enabled, geometry
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        db.execute(
            sql,
            (camera_id, config_version, resolution_width, resolution_height, roi_id, name, semantic_tag, enabled, geometry)
        )
        res = db.fetch_one("SELECT LAST_INSERT_ID()")
        return res[0] if res else None

    def get_roi(self, roi_id):
        sql = f"SELECT { self._fileds } FROM roi_configs WHERE roi_id=%s"
        return db.fetch_dict(sql, (roi_id,))

    def list_rois(self):
        sql = f"SELECT { self._fileds } FROM roi_configs ORDER BY id DESC"
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
        sql = f"UPDATE roi_configs SET {', '.join(fields)} WHERE roi_id=%s"
        params.append(roi_id)
        return db.execute(sql, tuple(params))

    def delete_roi(self, roi_id):
        sql = "DELETE FROM roi_configs WHERE roi_id=%s"
        return db.execute(sql, (roi_id,))

roi_config_db = RoiConfigDB()