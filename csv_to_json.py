import csv
import json
import os

def csv_to_json(csv_file, json_file):
    # Kiểm tra file tồn tại chưa
    if not os.path.exists(csv_file):
        print(f"❌ File không tồn tại: {csv_file}")
        return

    # Mở file CSV, loại bỏ BOM nếu có
    with open(csv_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Chuyển trường 'id' thành 'fruit_id' trong mỗi bản ghi
    for item in data:
        if "id" in item:
            item["fruit_id"] = item.pop("id")

    # Ghi ra file JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã chuyển {csv_file} sang {json_file} (đã loại BOM nếu có và đổi id thành fruit_id)")

if __name__ == "__main__":
    # Ví dụ đường dẫn — bạn thay theo đúng file của bạn
    csv_path = "data/metadata/fruit_metadata.csv"
    json_path = "data/metadata/fruit_metadata.json"
    csv_to_json(csv_path, json_path)
