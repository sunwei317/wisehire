#!/usr/bin/env python3
import subprocess
import sys

def setup_database():
    """设置数据库"""
    print("开始设置数据库...")
    
    # 尝试创建PostgreSQL数据库
    try:
        print("尝试创建PostgreSQL数据库...")
        result = subprocess.run([
            'sudo', '-u', 'postgres', 'createdb', 'resume_analysis'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL数据库 'resume_analysis' 创建成功")
            print("✅ 使用无密码PostgreSQL连接")
            return True
        else:
            print("❌ PostgreSQL数据库创建失败，错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 数据库设置异常: {e}")
        return False

def test_postgresql_connection():
    """测试PostgreSQL连接"""
    try:
        import psycopg2
        # 尝试无密码连接
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            database="resume_analysis",
            password="postgres"
        )
        conn.close()
        print("✅ PostgreSQL连接测试成功")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL连接失败: {e}")
        return False

if __name__ == "__main__":
    
    setup_database()
    test_postgresql_connection()
