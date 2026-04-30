import sys
import time
import os
import pandas as pd
from encode import Encode
from decode import Decode

def call_method(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name)
    return method(*args, **kwargs)

def compare_csv(file1, file2):
    # 读取 CSV 文件，强制使用整数格式，并移除空行
    #df1 = pd.read_csv(file1, dtype='int64').dropna().reset_index(drop=True)
    #df2 = pd.read_csv(file2, dtype='int64').dropna().reset_index(drop=True)
    df1 = pd.read_csv(file1, usecols=[0]).dropna().reset_index(drop=True)
    df2 = pd.read_csv(file2, usecols=[0]).dropna().reset_index(drop=True)

    # 转换为整数格式，移除空格和空行
    #df1 = df1.applymap(lambda x: int(str(x).strip()))
    #df2 = df2.applymap(lambda x: int(str(x).strip()))
    df1 = df1.applymap(lambda x: str(x).strip())
    df2 = df2.applymap(lambda x: str(x).strip())

    if df1.equals(df2):
        print("CSV file content is consistent")
    else:
        print("CSV file content is inconsistent")

    if df1.columns.equals(df2.columns):
        print("CSV file structure is consistent")
    else:
        print("CSV file structure is inconsistent")


def calculate_compression_ratio(in_file_path, out_file_path):
    import os
    original_size = os.path.getsize(in_file_path)
    compressed_size = os.path.getsize(out_file_path)

    if compressed_size == 0:
        print("Warning: Compressed file is empty. Cannot calculate compression ratio.")
        return

    compression_ratio = compressed_size / original_size
    compression_rate = (compression_ratio) * 100
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f} (Compressed/Original)")
    print(f"Compression rate: {compression_rate:.2f}%")


def main():
    if len(sys.argv) != 5 and len(sys.argv) != 4:
        print("Usage: program.py <en|de> <compression type> <data type> <file>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "diff":
        file_path1 = sys.argv[2]
        file_path2 = sys.argv[3]
        compare_csv(file_path1, file_path2)
    else:
        compression_type = sys.argv[2]
        data_type = sys.argv[3]
        in_file_path = sys.argv[4]

        # 创建 results 目录（如果不存在的话）
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)

        # 设置输出文件路径
        file_name = os.path.basename(in_file_path)
        out_file_path = os.path.join(output_dir, f"{file_name}.{compression_type if mode == 'en' else 'csv'}")
        method_name = mode + '_' + compression_type

        # 记录开始时间
        start_time = time.time()

        if mode == "en":
            en = Encode(compression_type, data_type, in_file_path, out_file_path)
            call_method(en, method_name)
            # 计算压缩率
            calculate_compression_ratio(in_file_path, out_file_path)
        elif mode == "de":
            de = Decode(compression_type, data_type, in_file_path, out_file_path)
            call_method(de, method_name)
        else:
            print("Invalid mode. Use 'en' for encoding or 'de' for decoding.")
            sys.exit(1)

        # 记录结束时间并计算运行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Output written to {out_file_path}")
        print(f"Execution time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()
