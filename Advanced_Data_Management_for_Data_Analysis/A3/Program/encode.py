import struct
import csv
import pickle
import json
class Encode:
    def __init__(self, compression_type, data_type, in_file_path, out_file_path):
        self.data_type = data_type
        self.in_file_path = in_file_path
        self.out_file_path = out_file_path
        self.compression_type = compression_type

    def en_bin(self):
        with open(self.in_file_path, 'r', newline='') as csvfile, open(self.out_file_path, 'wb') as binfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for item in row:
                    if self.data_type == 'string':
                        # 对字符串进行编码并写入二进制文件
                        encoded_str = item.encode('utf-8')
                        # 写入字符串长度（使用 int16 存储长度，最多支持 32767 字符）
                        binfile.write(struct.pack('h', len(encoded_str)))
                        # 写入字符串内容
                        binfile.write(encoded_str)
                    elif self.data_type == 'int8':
                        num = int(item)
                        binfile.write(struct.pack('b', num))
                    elif self.data_type == 'int16':
                        num = int(item)
                        binfile.write(struct.pack('h', num))
                    elif self.data_type == 'int32':
                        num = int(item)
                        binfile.write(struct.pack('i', num))
                    elif self.data_type == 'int64':
                        num = int(item)
                        binfile.write(struct.pack('q', num))
                    else:
                        raise ValueError("Unsupported data type.")

        print("Encoding completed for all columns.")

    def en_rle(self):

        def run_length_encoding(in_data):
            encoding = []
            i = 0
            while i < len(in_data):
                count = 1
                while i + 1 < len(in_data) and in_data[i] == in_data[i + 1]:
                    count += 1
                    i += 1
                encoding.append((in_data[i], count))
                i += 1
            return encoding

        data = []

        # 按文本模式读取文件内容
        with open(self.in_file_path, "r") as csvfile:
            for line in csvfile:
                line = line.strip()
                if self.data_type == 'string':
                    # 对于字符串类型，直接使用字符串值
                    value = line
                else:
                    # 对于整数类型，尝试转换为整数
                    try:
                        value = int(line)
                    except ValueError:
                        print(f"Error converting value: {line}")
                        continue
                data.append(value)

        encoded_data = run_length_encoding(data)

        # 写入编码后的数据到输出文件
        with open(self.out_file_path, "wb") as binfile:
            for value, count in encoded_data:
                if isinstance(value, str):
                    # 对于字符串类型，先写入字符串长度，再写入字符串和计数
                    encoded_value = value.encode('utf-8')
                    binfile.write(struct.pack('<I', len(encoded_value)))
                    binfile.write(encoded_value)
                    binfile.write(struct.pack('<q', count))
                else:
                    # 对于整数类型，直接写入值和计数
                    binfile.write(struct.pack('<q', value))
                    binfile.write(struct.pack('<q', count))

    def en_dic(self):
        def create_dictionary(data):
            unique_values = set(data)
            dictionary = {val: idx for idx, val in enumerate(unique_values)}
            return dictionary

        def encode_data(data, dictionary):
            return [dictionary[val] for val in data]

        def dictionary_encoding(input_file, output_file, data_type):
            with open(input_file, 'rb') as binfile:
                if data_type == 8:
                    data = struct.unpack('b' * (len(binfile.read())), binfile.read())
                elif data_type == 16:
                    data = struct.unpack('h' * (len(binfile.read()) // 2), binfile.read())
                elif data_type == 32:
                    data = struct.unpack('i' * (len(binfile.read()) // 4), binfile.read())
                elif data_type == 64:
                    data = struct.unpack('q' * (len(binfile.read()) // 8), binfile.read())
                else:
                    data = binfile.read().decode('utf-8')

            dictionary = create_dictionary(data)

            encoded_data = encode_data(data, dictionary)

            with open(output_file, 'wb') as outfile:
                pickle.dump((dictionary, encoded_data), outfile)

        dictionary_encoding(self.in_file_path, self.out_file_path, self.data_type)

    def en_for(self, block_size=128):
        data = []
        string_dict = {}
        string_counter = 0

        # 从 CSV 文件中读取数据并转换为整数格式
        with open(self.in_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 0:
                    try:
                        # 尝试将值转换为整数
                        value = int(row[0])
                        data.append(value)
                    except ValueError:
                        # 如果转换失败，认为是字符串，进行字典编码
                        string_value = row[0]
                        if string_value not in string_dict:
                            string_dict[string_value] = string_counter
                            string_counter += 1
                        encoded_value = string_dict[string_value]
                        data.append(encoded_value)

        if not data:
            print("Input file is empty or data conversion failed.")
            return

        # 保存字符串字典到 JSON 文件
        dict_file_path = self.out_file_path + ".dict.json"
        with open(dict_file_path, 'w') as dict_file:
            json.dump(string_dict, dict_file)
        print(f"String dictionary saved to: {dict_file_path}")

        # 编码并写入二进制文件
        with open(self.out_file_path, 'wb') as outfile:
            index = 0
            while index < len(data):
                # 获取当前数据块
                block = data[index:index + block_size]

                # 找到参考值
                reference = min(block)
                offsets = [value - reference for value in block]

                # 写入参考值
                outfile.write(struct.pack('<i', reference))

                # 写入偏移量
                for offset in offsets:
                    if -2147483648 <= offset <= 2147483647:  # 检查偏移量是否在 int32 范围内
                        outfile.write(struct.pack('<i', offset))
                    else:
                        print(f"Warning: Offset {offset} out of range for int32. Data may be corrupted.")
                        outfile.write(struct.pack('<i', 0))  # 写入 0 作为占位符

                index += block_size

        print("Encoding completed with string dictionary.")

    def en_dif(self):

        if self.data_type == 'string':
            print(
                "Warning: 'dif' encoding is not suitable for string data. Please use 'bin', 'rle', or 'for' encoding instead.")
            return
        data = []
        # 判断文件格式是否为 CSV 文本文件
        try:
            # 尝试以文本格式读取 CSV 文件
            with open(self.in_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) > 0:
                        try:
                            value = int(row[0])
                            data.append(value)
                        except ValueError:
                            print(f"Error converting value: {row[0]}")
                            continue

            if not data:
                print("Input file is empty or data conversion failed.")
                return
        except Exception as e:
            print("Assuming binary format due to error:", e)
            # 如果文件不是 CSV 格式，按二进制格式读取
            with open(self.in_file_path, 'rb') as binfile:
                while True:
                    if self.data_type == 'int8':
                        chunk = binfile.read(1)
                        if len(chunk) < 1:
                            break
                        data.append(struct.unpack('b', chunk)[0])
                    elif self.data_type == 'int16':
                        chunk = binfile.read(2)
                        if len(chunk) < 2:
                            break
                        data.append(struct.unpack('h', chunk)[0])
                    elif self.data_type == 'int32':
                        chunk = binfile.read(4)
                        if len(chunk) < 4:
                            break
                        data.append(struct.unpack('i', chunk)[0])
                    elif self.data_type == 'int64':
                        chunk = binfile.read(8)
                        if len(chunk) < 8:
                            break
                        data.append(struct.unpack('q', chunk)[0])
                    else:
                        print(f"Unsupported data type: {self.data_type}")
                        return

        # 计算并写入差分编码
        with open(self.out_file_path, 'wb') as outfile:
            previous_value = 0
            for current_value in data:
                difference = current_value - previous_value
                previous_value = current_value

                if self.data_type == 'int8':
                    outfile.write(struct.pack('b', difference))
                elif self.data_type == 'int16':
                    outfile.write(struct.pack('h', difference))
                elif self.data_type == 'int32':
                    outfile.write(struct.pack('i', difference))
                elif self.data_type == 'int64':
                    outfile.write(struct.pack('q', difference))