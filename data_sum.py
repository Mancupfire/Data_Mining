import os
import json
import numpy as np
import random

def show_data_samples(dataset_name="scientific", data_dir="./dataset"):
    dataset_path = os.path.join(data_dir, dataset_name)
    train_file = os.path.join(dataset_path, f"{dataset_name}.train.jsonl")
    map_file = os.path.join(dataset_path, f"{dataset_name}.emb_map.json")
    feature_file = os.path.join(dataset_path, "item_features.npy")

    # 1. Load Item Map & Features
    with open(map_file, 'r') as f:
        item2id = json.load(f)
    
    if not os.path.exists(feature_file):
        print(f"Không tìm thấy file {feature_file}")
        return
    features = np.load(feature_file)

    # 2. Lấy ngẫu nhiên 1 dòng dữ liệu từ tập Train
    with open(train_file, 'r') as f:
        lines = f.readlines()
        sample_line = json.loads(random.choice(lines)) # Bốc random 1 user

    user_id = sample_line.get("user_id", "Unknown")
    history_tokens = sample_line.get("inter_history", [])
    target_token = sample_line.get("target_id", "")

    # Map sang dạng ID số nguyên để mô hình đọc
    history_ids = [item2id[tok] for tok in history_tokens if tok in item2id]
    target_id = item2id.get(target_token, -1)

    # 3. Trình bày ra màn hình
    print("\n" + "="*70)
    print(" 🔍 TRỰC QUAN HÓA DỮ LIỆU HYBRID (SEQUENCE + FEATURE) 🔍")
    print("="*70)

    print("\n[1] SEQUENTIAL DATA (Dữ liệu Tuần tự - Lịch sử User)")
    print("------------------------------------------------------")
    print(f"👤 User ID      : {user_id}")
    print(f"🛒 Raw Tokens   : {history_tokens[:5]}{' ...' if len(history_tokens)>5 else ''} (Độ dài: {len(history_tokens)})")
    print(f"🔢 Mapped IDs   : {history_ids[:5]}{' ...' if len(history_ids)>5 else ''}")
    print(f"🎯 Target Item  : {target_token} (Mapped ID: {target_id})")
    print("-> T5 Encoder sẽ học sự chuyển dịch theo thời gian của chuỗi ID này.")

    print("\n[2] FEATURE DATA (Dữ liệu Đặc trưng - Thông tin Item)")
    print("------------------------------------------------------")
    # Lấy item đầu tiên trong lịch sử để minh họa
    sample_item_id = history_ids[0] if history_ids else target_id
    sample_item_token = history_tokens[0] if history_tokens else target_token
    sample_feature = features[sample_item_id]

    print(f"📦 Chọn 1 Item  : {sample_item_token} (ID: {sample_item_id})")
    print(f"📏 Feature Shape: {sample_feature.shape} (256 chiều)")
    
    # Format vector cho đẹp (chỉ in 8 giá trị đầu và 2 giá trị cuối)
    vec_start = ", ".join([f"{x: .4f}" for x in sample_feature[:8]])
    vec_end = ", ".join([f"{x: .4f}" for x in sample_feature[-2:]])
    print(f"🧬 Vector Values: [{vec_start}, ..., {vec_end}]")
    print("-> Đây là tri thức ngữ nghĩa (Semantic) hoặc Multi-field của item.")

    print("\n[3] SỰ KẾT HỢP (HYBRID INTERACTION)")
    print("------------------------------------------------------")
    print("Thay vì chỉ nhìn vào chuỗi ID khô khan, T5 sẽ lấy từng ID (ví dụ: ID", sample_item_id, ")")
    print("sau đó 'nhúng' trực tiếp vector đặc trưng 256-dim của nó vào luồng Attention.")
    print("=> Mô hình vừa hiểu 'Ngữ cảnh lịch sử', vừa hiểu 'Bản chất món hàng'!")
    print("="*70 + "\n")

if __name__ == "__main__":
    show_data_samples("scientific")