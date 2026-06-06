import os
import numpy as np
import json

dataset_dir = "./dataset/scientific"
input_emb_path = os.path.join(dataset_dir, "scientific_emb_256.npy")
output_feature_path = os.path.join(dataset_dir, "item_features.npy")
map_path = os.path.join(dataset_dir, "scientific.emb_map.json")

# 1. Load ma trận embedding gốc
# Kích thước đang là: (num_items, 256)
original_emb = np.load(input_emb_path)

# 2. Tạo một vector padding toàn số 0 (kích thước: 1 x 256)
padding_row = np.zeros((1, original_emb.shape[1]), dtype=np.float32)

# 3. Nối (concatenate) vector 0 lên đầu ma trận gốc
# Kích thước mới sẽ là: (num_items + 1, 256)
feature_matrix = np.concatenate([padding_row, original_emb], axis=0)

# 4. Lưu ra file item_features.npy
np.save(output_feature_path, feature_matrix)

print(f"Đã tạo thành công {output_feature_path}!")
print(f"Kích thước ma trận tính cả padding: {feature_matrix.shape}")

# Kiểm tra chéo với file map (tùy chọn)
with open(map_path, 'r') as f:
    item2id = json.load(f)
print(f"Tổng số items trong map: {len(item2id)}")
if feature_matrix.shape[0] == len(item2id) + 1:
    print("-> CHUẨN! Kích thước ma trận đã khớp hoàn hảo với ID.")