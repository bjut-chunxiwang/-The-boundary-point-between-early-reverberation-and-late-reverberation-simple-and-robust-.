import pandas as pd
import matplotlib.pyplot as plt

# 文件路径和列名的定义
file_path = r"C:\Users\Lenovo\Desktop\simple&roubust\read(BP).csv"  # 替换为你的 CSV 文件路径
column_name = "t_abel"  # 替换为你需要绘制的列名

# 读取 CSV 文件
try:
    data = pd.read_csv(file_path)
    if column_name not in data.columns:
        print(f"列 {column_name} 不存在，请检查文件列名。")
    else:
        # 获取指定列的数据
        column_data = data[column_name].dropna()  # 移除缺失值

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(column_data, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {column_name}', fontsize=16)
        plt.xlabel(column_name, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查文件路径。")
except Exception as e:
    print(f"出现错误：{e}")
