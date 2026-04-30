import struct
import csv
import pickle
import json

class Decode:
    def __init__(self, compression_type, data_type, in_file_path, out_file_path):
        self.data_type = data_type
        self.in_file_path = in_file_path
        self.out_file_path = out_file_path
        self.compression_type = compression_type

    def de_bin(self, original_file_path=None):
        # 检查并初始化 num_columns 属性
        if not hasattr(self, 'num_columns'):
            if original_file_path:
                try:
                    with open(original_file_path, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        self.num_columns = len(next(reader))
                        print(f"Number of columns detected: {self.num_columns}")
                except Exception as e:
                    print(f"Error reading original file for column count: {e}")
                    self.num_columns = 1  # 默认设置为 1 列
            else:
                self.num_columns = 1

        with open(self.in_file_path, 'rb') as binfile, open(self.out_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = []

            while True:
                try:
                    if self.data_type == 'string':
                        # 读取字符串长度（int16）
                        length_data = binfile.read(2)
                        if not length_data:
                            break
                        str_len = struct.unpack('h', length_data)[0]
                        # 读取字符串内容
                        str_data = binfile.read(str_len)
                        if len(str_data) < str_len:
                            print("End of file or incomplete data block (string).")
                            break
                        decoded_str = str_data.decode('utf-8')
                        row.append(decoded_str)
                    elif self.data_type == 'int8':
                        data = binfile.read(1)
                        if not data:
                            break
                        num = struct.unpack('b', data)[0]
                        row.append(num)
                    elif self.data_type == 'int16':
                        data = binfile.read(2)
                        if not data:
                            break
                        num = struct.unpack('h', data)[0]
                        row.append(num)
                    elif self.data_type == 'int32':
                        data = binfile.read(4)
                        if not data:
                            break
                        num = struct.unpack('i', data)[0]
                        row.append(num)
                    elif self.data_type == 'int64':
                        data = binfile.read(8)
                        if not data:
                            break
                        num = struct.unpack('q', data)[0]
                        row.append(num)
                    else:
                        raise ValueError("Unsupported data type.")

                    # 当读到一整行数据时，写入 CSV 文件
                    if len(row) == self.num_columns:
                        writer.writerow(row)
                        row = []

                except Exception as e:
                    print(f"Error during decoding: {e}")
                    break

            print("Decoding completed for all columns.")

    def de_rle(self):

        def rle_decode(encoded_data):
            decoded_data = []
            for value, count in encoded_data:
                decoded_data.extend([value] * count)
            return decoded_data

        encoded_data = []

        with open(self.in_file_path, "rb") as binfile:
            while True:
                try:
                    if self.data_type == 'string':
                        # 读取字符串长度
                        length_chunk = binfile.read(4)
                        if len(length_chunk) < 4:
                            break
                        length = struct.unpack('<I', length_chunk)[0]

                        # 读取字符串值
                        value_chunk = binfile.read(length)
                        if len(value_chunk) < length:
                            break
                        value = value_chunk.decode('utf-8')

                        # 读取计数
                        count_chunk = binfile.read(8)
                        if len(count_chunk) < 8:
                            break
                        count = struct.unpack('<q', count_chunk)[0]
                    else:
                        # 对于整数类型，读取值和计数
                        value_chunk = binfile.read(8)
                        count_chunk = binfile.read(8)
                        if len(value_chunk) < 8 or len(count_chunk) < 8:
                            break
                        value = struct.unpack('<q', value_chunk)[0]
                        count = struct.unpack('<q', count_chunk)[0]

                    encoded_data.append((value, count))
                except Exception as e:
                    print(f"Error during decoding: {e}")
                    break

        decoded_data = rle_decode(encoded_data)

        # 写入解码后的值到 CSV 文件
        with open(self.out_file_path, "w", newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            for value in decoded_data:
                writer.writerow([value])

    def de_dic(self):
        def decode_data(encoded_data, dictionary):
            reverse_dictionary = {v: k for k, v in dictionary.items()}
            return [reverse_dictionary[val] for val in encoded_data]

        def dictionary_decoding(input_file, output_file, data_type):
            with open(input_file, 'rb') as infile:
                dictionary, encoded_data = pickle.load(infile)

            decoded_data = decode_data(encoded_data, dictionary)

            with open(output_file, 'wb') as outfile:
                if data_type == 8:
                    outfile.write(struct.pack('b' * len(decoded_data), *decoded_data))
                elif data_type == 16:
                    outfile.write(struct.pack('h' * len(decoded_data), *decoded_data))
                elif data_type == 32:
                    outfile.write(struct.pack('i' * len(decoded_data), *decoded_data))
                elif data_type == 64:
                    outfile.write(struct.pack('q' * len(decoded_data), *decoded_data))
                else:
                    outfile.write(''.join(decoded_data).encode('utf-8'))

        dictionary_decoding(self.in_file_path, self.out_file_path, self.data_type)

    def de_for(self, block_size=128):
        dict_file_path = self.in_file_path + ".dict.json"
        reverse_dict = {}

        # 加载字符串字典
        try:
            with open(dict_file_path, 'r') as dict_file:
                string_dict = json.load(dict_file)
                reverse_dict = {v: k for k, v in string_dict.items()}
            print(f"String dictionary loaded from: {dict_file_path}")
        except FileNotFoundError:
            print(
                f"Warning: String dictionary file not found at {dict_file_path}. Decoding may be incorrect for string data.")

        with open(self.in_file_path, 'rb') as infile, open(self.out_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile, lineterminator='\n')

            while True:
                reference_data = infile.read(4)
                if len(reference_data) < 4:
                    break
                reference = struct.unpack('<i', reference_data)[0]
                offsets_data = infile.read(block_size * 4)
                if len(offsets_data) % 4 != 0:
                    print("Warning: Incomplete data block at the end of the file.")
                    offsets_data = offsets_data[:len(offsets_data) - (len(offsets_data) % 4)]

                offsets = list(struct.unpack(f'<{len(offsets_data) // 4}i', offsets_data))
                decoded_data = [reference + offset for offset in offsets]

                for value in decoded_data:
                    if value in reverse_dict:
                        writer.writerow([reverse_dict[value]])
                    else:
                        writer.writerow([value])

        print("Decoding completed.")

    def de_dif(self):
        with open(self.in_file_path, 'rb') as infile, open(self.out_file_path, 'w', newline='') as outfile:
            import csv
            writer = csv.writer(outfile)
            previous_value = 0

            while True:
                difference = None  # 初始化 difference 变量
                if self.data_type == "int8":
                    data = infile.read(1)
                    if len(data) < 1:
                        print("End of file or incomplete data block (int8).")
                        break
                    difference = struct.unpack('b', data)[0]
                elif self.data_type == "int16":
                    data = infile.read(2)
                    if len(data) < 2:
                        print("End of file or incomplete data block (int16).")
                        break
                    difference = struct.unpack('h', data)[0]
                elif self.data_type == "int32":
                    data = infile.read(4)
                    if len(data) < 4:
                        print("End of file or incomplete data block (int32).")
                        break
                    difference = struct.unpack('i', data)[0]
                elif self.data_type == "int64":
                    data = infile.read(8)
                    if len(data) < 8:
                        print("End of file or incomplete data block (int64).")
                        break
                    difference = struct.unpack('q', data)[0]
                else:
                    print(f"Unsupported data type: {self.data_type}")
                    break

                # 确保 difference 已被正确赋值
                if difference is None:
                    print("Error: difference not assigned.")
                    break

                # 计算原始值并更新 previous_value
                current_value = previous_value + difference
                previous_value = current_value

                # 写入解码后的值到 CSV 文件
                writer.writerow([current_value])

            print("Decoding completed.")






