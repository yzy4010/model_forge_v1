import pymysql.cursors
from dbutils.pooled_db import PooledDB
import pandas as pd
import numpy as np

# 数据库连接配置
# host = 'localhost'
# port = 3306
# user = 'root'
# password = '123456'
# database = 'deepcarft'

# host = '121.237.177.224'
# port = 3306
# user = 'eims'
# password = 'Dcom123456'
# database = 'actioncheck_ai'

host = '121.237.177.224'
port = 3306
user = 'eims'
password = 'Dcom123456'
database = 'fjy-ai-177'

# 创建连接池
pool = PooledDB(
    creator=pymysql, # 使用链接数据库的模块
    mincached=10, # 初始化时，链接池中至少创建的链接，0表示不创建
    maxconnections=200, # 连接池允许的最大连接数，0和None表示不限制连接数
    blocking=True, # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
    host=host,
    port=port,
    user=user,
    password=password,
    database=database
)


class DBHelper:
    """
    数据库操作工具类，支持增删改查、批量插入和事务处理，带分页方法
    """

    def __init__(self):
        # 使用连接池获取连接
        self.pool = pool

    def get_conn(self):
        """ 获取连接和游标 """
        conn = self.pool.connection()
        cursor = conn.cursor()
        return conn, cursor

    def query(self, sql, params=None):
        """
        执行单条查询
        :param sql: 查询语句
        :param params: 可选参数
        :return: 查询结果
        """
        conn, cursor = self.get_conn()
        try:
            cursor.execute(sql, params if params else ())
            res = cursor.fetchall()
        except Exception as e:
            print(f'查询出错: {e}')
            res = None
        finally:
            cursor.close()
            conn.close()
        return res

    def execute(self, sql, params=None):
        """
        执行单条增删改语句
        :param sql: SQL 语句
        :param params: 可选参数
        :return: 受影响的行数
        """
        conn, cursor = self.get_conn()
        rowcount = 0
        try:
            rowcount = cursor.execute(sql, params if params else ())
            conn.commit()
        except Exception as e:
            print(f'执行出错: {e}')
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
        return rowcount

    def batch_insert(self, sql, data):
        """
        批量插入数据
        :param sql: 形如 INSERT INTO table (col1, col2) VALUES (%s, %s)
        :param data: 列表嵌套元组 [[a, b], [c, d], ...]
        :return: 插入的行数
        """
        conn, cursor = self.get_conn()
        rowcount = 0
        try:
            rowcount = cursor.executemany(sql, data)
            conn.commit()
        except Exception as e:
            print(f'批量插入出错: {e}')
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
        return rowcount

    def execute_many(self, sql, data):
        """
        通用批量执行（如批量删除、更新），同 executemany
        :param sql: 形如 UPDATE xxx SET ... WHERE ...  或 DELETE FROM ...
        :param data: [(params1...), (params2...), ...]
        """
        return self.batch_insert(sql, data)

    def execute_in_transaction(self, actions):
        """
        执行带有事务的多步数据库操作
        :param actions: [(sql1, params1), (sql2, params2), ...]
        :return: True 成功，False 失败
        """
        conn, cursor = self.get_conn()
        success = True
        try:
            for sql, params in actions:
                cursor.execute(sql, params if params else ())
            conn.commit()
        except Exception as e:
            print(f'事务操作出错: {e}')
            conn.rollback()
            success = False
        finally:
            cursor.close()
            conn.close()
        return success

    def fetch_one(self, sql, params=None):
        """
        查询并只取第一条
        """
        conn, cursor = self.get_conn()
        try:
            cursor.execute(sql, params if params else ())
            res = cursor.fetchone()
        except Exception as e:
            print(f'查询出错: {e}')
            res = None
        finally:
            cursor.close()
            conn.close()
        return res

    def fetch_dict(self, sql, params=None):
        """
        查询结果为字典形式
        """
        conn = self.pool.connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute(sql, params if params else ())
            res = cursor.fetchall()
        except Exception as e:
            print(f'查询出错: {e}')
            res = None
        finally:
            cursor.close()
            conn.close()
        return res

    def fetch_dict_page(self, sql, params=None, page=1, page_size=20):
        """
        分页查询，返回数据（字典格式）、总条数与总页数
        :param sql: 查询SQL（不能带LIMIT和OFFSET）
        :param params: 可选参数
        :param page: 当前页（1开始）
        :param page_size: 每页条数
        :return: { "data": [...], "total": 总条数, "total_pages": 总页数, "page": 当前页 }
        """
        # 1. 总数
        count_sql = f"SELECT COUNT(*) as count FROM ({sql}) t"
        total = 0
        conn = self.pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(count_sql, params if params else ())
            total = cursor.fetchone()[0]
        except Exception as e:
            print(f'分页计数出错: {e}')
            total = 0
        finally:
            cursor.close()
            conn.close()

        # 2. 分页数据
        offset = (page - 1) * page_size
        page_sql = f"{sql} LIMIT %s OFFSET %s"
        query_params = list(params) if params else []
        query_params.extend([page_size, offset])

        conn = self.pool.connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        try:
            cursor.execute(page_sql, tuple(query_params))
            data = cursor.fetchall()
        except Exception as e:
            print(f'分页查询出错: {e}')
            data = None
        finally:
            cursor.close()
            conn.close()

        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

        return {
            "data": data,
            "total": total,
            "total_pages": total_pages,
            "page": page
        }

    def fetch_page(self, sql, params=None, page=1, page_size=20):
        """
        分页查询，普通fetchall（非字典格式）
        :param sql: 查询SQL（不能带LIMIT和OFFSET）
        :param params: 可选参数
        :param page: 当前页（1开始）
        :param page_size: 每页条数
        :return: { "data": [...], "total": 总条数, "total_pages": 总页数, "page": 当前页 }
        """
        count_sql = f"SELECT COUNT(*) as count FROM ({sql}) t"
        total = 0
        conn = self.pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(count_sql, params if params else ())
            total = cursor.fetchone()[0]
        except Exception as e:
            print(f'分页计数出错: {e}')
            total = 0
        finally:
            cursor.close()
            conn.close()

        offset = (page - 1) * page_size
        page_sql = f"{sql} LIMIT %s OFFSET %s"
        query_params = list(params) if params else []
        query_params.extend([page_size, offset])

        conn = self.pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(page_sql, tuple(query_params))
            data = cursor.fetchall()
        except Exception as e:
            print(f'分页查询出错: {e}')
            data = None
        finally:
            cursor.close()
            conn.close()

        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

        return {
            "data": data,
            "total": total,
            "total_pages": total_pages,
            "page": page
        }


db = DBHelper()