import numpy as np

def calculate_dot_product(byte_a, byte_b):
    """计算两个 8-bit (4个三值数) 的点积"""
    dot_product = 0
    ternary_map = {0: 0, 1: 1, 2: -1, 3: 0} # 00->0, 01->1, 10->-1, 11->0 (invalid)

    for i in range(4): # 一个 byte 里有 4 个 2-bit 数
        bits_a = (byte_a >> (i * 2)) & 0b11
        bits_b = (byte_b >> (i * 2)) & 0b11
        
        val_a = ternary_map[bits_a]
        val_b = ternary_map[bits_b]
        
        dot_product += val_a * val_b
        
    return dot_product

# LUT 大小为 256x256，因为一个 byte 有 256 种可能
LUT_SIZE = 256
lut = np.zeros((LUT_SIZE, LUT_SIZE), dtype=np.int8)

for i in range(LUT_SIZE):
    for j in range(LUT_SIZE):
        lut[i, j] = calculate_dot_product(i, j)

# 保存 LUT 到二进制文件
lut.tofile('ternary_lut.bin')

print(f"查找表已生成: ternary_lut.bin (大小: {LUT_SIZE*LUT_SIZE} bytes)")