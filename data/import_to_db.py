import sqlite3
import pandas as pd
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# CSV文件路径
csv_file_path = os.path.join(script_dir, "raw", "noaa_weather_data.csv")

# 数据库文件路径（在data目录下创建）
db_file_path = os.path.join(script_dir, "weather_data.db")

def main():
    print(f"开始将CSV数据导入数据库...")
    print(f"CSV文件路径: {csv_file_path}")
    print(f"数据库文件路径: {db_file_path}")
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件 {csv_file_path} 不存在")
        return
    
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(csv_file_path)
        
        # 显示数据信息
        print(f"CSV文件读取成功，共有 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        print("前5行数据:")
        print(df.head())
        
        # 连接到SQLite数据库（如果不存在则创建）
        print("\n正在连接数据库...")
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        
        # 创建表（如果不存在）
        print("正在创建表...")
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS noaa_weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            时间 TEXT,
            降雨量 REAL,
            持续时间 INTEGER,
            风速 REAL,
            最低温度 REAL,
            最高温度 REAL,
            平均温度 REAL,
            洪涝风险标签 INTEGER
        )
        '''
        cursor.execute(create_table_query)
        
        # 删除表中现有数据（可选）
        print("正在清空表中现有数据...")
        cursor.execute("DELETE FROM noaa_weather_data")
        
        # 将数据插入表中
        print("正在将数据插入数据库...")
        # 准备插入语句
        insert_query = '''
        INSERT INTO noaa_weather_data (
            时间, 降雨量, 持续时间, 风速, 最低温度, 最高温度, 平均温度, 洪涝风险标签
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        # 将DataFrame转换为元组列表
        data_tuples = [tuple(x) for x in df.to_numpy()]
        
        # 执行批量插入
        cursor.executemany(insert_query, data_tuples)
        
        # 提交事务
        conn.commit()
        
        # 验证数据是否正确导入
        print("正在验证数据导入...")
        cursor.execute("SELECT COUNT(*) FROM noaa_weather_data")
        count = cursor.fetchone()[0]
        print(f"数据库中共有 {count} 条记录")
        
        # 显示前5条记录
        print("数据库中前5条记录:")
        cursor.execute("SELECT * FROM noaa_weather_data LIMIT 5")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        
        print("\n数据导入完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        # 关闭数据库连接
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
