import pandas as pd

# 文件路径
file_path = r"C:\Users\Lenovo\Desktop\simple&roubust\read(BP).csv"  # 替换为你的 CSV 文件路径

# 读取 CSV 文件
try:
    # 加载数据
    data = pd.read_csv(file_path)

    # 检查是否存在列 't_abel'
    if 't_abel' not in data.columns:
        print(f"列 't_abel' 不存在，请检查文件列名。")
    else:
        # 进行数值处理：乘以48后取整，创建新列 'BP2'
        data['BP2'] = (data['t_abel'] * 48).round(0).astype(int)

        # 保存结果到新文件或查看结果
        output_file = "output_with_BP2.csv"  # 输出文件名
        data.to_csv(output_file, index=False)
        print(f"新文件已保存到 {output_file}")

except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查文件路径。")
except Exception as e:
    print(f"出现错误：{e}")
