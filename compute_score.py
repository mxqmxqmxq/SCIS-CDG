import os
import subprocess

# 设置网络的基础目录和输出目录
base_dir = '/root/autodl-tmp/HGDC-master/data/cancer_Net'
output_base_dir = '/root/rlap-main_fe_gnn/cancer_Net'
script_path = '/root/rlap-main_fe_gnn/node_shared_PGDC.py'

# 获取所有网络的目录名称
networks = [dir for dir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dir))]

# 遍历每个网络目录
for network in networks:
    network_dir = os.path.join(base_dir, network)
    # 获取该网络目录下的所有.pkl文件
    cancer_types = [f for f in os.listdir(network_dir) if f.endswith('.pkl')]

    # 遍历每种癌症类型，为每个.pkl文件执行脚本
    for pkl_file in cancer_types:
        cancer_type = pkl_file.split('_')[1]
        dataset_file_path = os.path.join(network_dir, pkl_file)
        output_dir = os.path.join(output_base_dir, network, 'Result')
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f'predicted_scores_{cancer_type}_{network}.txt')

        # 构建运行命令
        command = f'python3 {script_path} --dataset_file {dataset_file_path} --output {output_file_path}'

        # 执行命令
        subprocess.run(command, shell=True)

        print(f'已完成 {cancer_type} 在网络 {network} 的处理')
