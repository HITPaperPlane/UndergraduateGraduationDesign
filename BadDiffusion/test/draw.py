import os
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip lines that don't have enough data
        label = int(parts[0])
        values = list(map(float, parts[1:5]))
        data.append((label, values))
    return data

def process_files(file_paths):
    all_data = []
    max_length = 0
    for file_path in file_paths:
        data = read_file(file_path)
        all_data.append(data)
        if len(data) > max_length:
            max_length = len(data)
    
    # Pad shorter files with None
    for i in range(len(all_data)):
        if len(all_data[i]) < max_length:
            all_data[i] += [(None, [None]*4)] * (max_length - len(all_data[i]))
    
    return all_data

def plot_data(all_data, file_names):
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(all_data):
        labels = []
        max_values = []
        mean_values = []
        for label, values in data:
            if label is None:
                labels.append(None)
                max_values.append(None)
                mean_values.append(None)
            else:
                labels.append(label)
                max_values.append(max(values))
                mean_values.append(np.mean(values))
        
        plt.plot(labels, max_values, label=f"{file_names[i]} (Max)", marker='o')
        plt.plot(labels, mean_values, label=f"{file_names[i]} (Mean)", marker='x')
    
    plt.xlabel('Label')
    plt.ylabel('Value')
    plt.title('Max and Mean Values for Each File')
    plt.legend()
    plt.grid(True)
    # 保存图形为图片
    output_file = 'output_plot.png'  # 设置保存的文件名和路径
    plt.savefig(output_file)

    plt.show()

def main():
    # List of txt files to process
    file_paths = ['test/sim/clnorm.txt', 'test/sim/normal.txt']  # Replace with your actual file paths
    file_names = [os.path.basename(path) for path in file_paths]
    
    all_data = process_files(file_paths)
    plot_data(all_data, file_names)

if __name__ == "__main__":
    main()