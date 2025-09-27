from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import glob
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    email = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisTask(Base):
    __tablename__ = 'analysis_tasks'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    resume_filename = Column(String(255), nullable=False)
    job_description = Column(Text, nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='in_process')
    result_json = Column(Text)
    updated_at = Column(DateTime)

# 数据库配置 - 使用密码
DATABASE_URL = "postgresql://postgres:postgres@localhost/resume_analysis"

try:
    engine = create_engine(DATABASE_URL)
    # 测试连接
    with engine.connect() as conn:
        print("✅ PostgreSQL数据库连接成功")
except Exception as e:
    print(f"❌ PostgreSQL连接失败: {e}")
    # 使用SQLite作为备用
    DATABASE_URL = "sqlite:///./resume_analysis.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print("✅ 使用SQLite作为备用数据库")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 数据库表创建成功")
        
        # 插入示例数据
        db = SessionLocal()
        try:
            # 检查是否已有用户
            if not db.query(User).first():
                users = [
                    User(username="admin", password="password123", email="admin@example.com"),
                    User(username="user", password="user123", email="user@example.com")
                ]
                db.add_all(users)
                db.commit()
                print("✅ 示例用户创建成功")
        except Exception as e:
            print(f"创建用户时出错: {e}")
            db.rollback()
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ 创建表失败: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

import psycopg2
from datetime import datetime

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'database': 'resume_analysis',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

def create_task(task):
    """创建新任务"""

    try:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # SQL 插入语句
        insert_query = """
        INSERT INTO analysis_tasks (user_id, resume_filename, job_description, submitted_at, status,result_json,updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        # 执行插入
        cursor.execute(insert_query, (
            task.get("user_id"), 
            task.get("resume_filename"), 
            task.get("job_description"), 
            task.get("submitted_at"), 
            task.get("status"),
            task.get("result_json"),
            task.get("updated_at")
        ))
        
        # 获取新创建的任务ID
        task_id = cursor.fetchone()[0]
        
        # 提交事务
        conn.commit()
        print(f"任务创建成功！ID: {task_id}")
        
        return task_id
        
    except Exception as e:
        print(f"创建任务失败: {e}")
        conn.rollback()
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def update_task(task):
    """update任务"""

    try:
        # 连接数据库
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # SQL 插入语句
        
        update_query = """
        UPDATE analysis_tasks 
        SET status =  %(status)s , result_json = %(result_json)s, updated_at=%(updated_at)s
        WHERE id = %(task_id)s;
        """

        cursor.execute(
            update_query,
            {
                "task_id": task.get("task_id"),
                "status": task.get("status"),
                "result_json": task.get("result_json"),
                "updated_at": task.get("updated_at"),
            },
        )
        # 提交事务（每个文件插入完成后提交一次）
        conn.commit()

    except Exception as e:
        print(f"update failed: {e}")
        conn.rollback()
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def delete_task_from_db(task_id, user_id):
    
        to_be_deleted_report=os.path.join("analysis_reports",f"task_id_{task_id}_user_{user_id}.json")
        if os.path.exists(to_be_deleted_report):
            os.remove(to_be_deleted_report)


        to_be_deleted_resume = os.path.join("static/uploads",f"task_id_{task_id}_user_{user_id}.*")
        
        files = glob.glob(to_be_deleted_resume)
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
                    

        # 如果环境变量不存在，使用默认连接参数
        conn = psycopg2.connect(
                host='localhost',
                database='resume_analysis',
                user='postgres',
                password='postgres',
                port=5432
        )
    
        # 创建游标
        cur = conn.cursor()
        
        # 删除数据库记录
        cur.execute("""
            DELETE FROM analysis_tasks 
            WHERE id = %s AND user_id = %s
        """, (task_id, user_id))
        
        # 提交事务
        conn.commit()
        conn.close()
        return {"message": "Task deleted successfully"}
        