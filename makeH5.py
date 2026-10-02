import pandas as pd
import os

# ================== 配置区 ==================
csv_file = '/home/yanke/TDL/GSE87375/raw/cleanedGSE87375_Single_Cell_RNA-seq_Gene_TPM.csv'  # 输入文件
output_dir = '/home/yanke/TDL/GSE87375/raw/mouse_tpm'  # 输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 时间点列表（按照你的描述顺序）
time_points = [ 'P0', 'P3', 'P9', 'P15', 'P18', 'P60']

# ================== 读取数据 ==================
df = pd.read_csv(csv_file)

# 将 Symbol 列统一为小写
df['Symbol'] = df['Symbol'].str.strip().str.lower()

# 输出 α/β 统计
cell_type_counts = {}

# ================== 生成 h5 文件 ==================
for i, tp in enumerate(time_points):
    # 根据列名包含时间点筛选
    cols_tp = [col for col in df.columns if tp in col]
    if len(cols_tp) == 0:
        print(f'Warning: {tp} columns not found!')
        continue

    df_tp = df[['Symbol'] + cols_tp].copy()

    # 统计 alpha / beta
    alpha_count = sum(['a' in col.lower() for col in cols_tp])
    beta_count  = sum(['b' in col.lower() for col in cols_tp])
    cell_type_counts[tp] = {'alpha': alpha_count, 'beta': beta_count}

    # 保存 h5 文件，key = 'RPKMs'
    h5_file = os.path.join(output_dir, f'mouse_tpm_{i}.h5')
    store = pd.HDFStore(h5_file)
    store.put('RPKMs', df_tp)
    store.close()
    print(f'{tp} h5 file saved: {h5_file}, alpha={alpha_count}, beta={beta_count}')

print("All files generated.")
print("α/β counts per time point:")
for tp, counts in cell_type_counts.items():
    print(f'{tp}: {counts}')
