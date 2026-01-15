from flask import Flask, render_template, request, flash, jsonify
from qdrant_utils.search_text_vectors import get_text_vector, search_by_vector
from qdrant_client import QdrantClient
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import torch

#khởi tạo flask
app = Flask(__name__, static_folder='static')
app.secret_key = "your_secret_key_here"

# Kết nối Qdrant
client = QdrantClient("http://localhost", port=6333)

# Load model EfficientNet-B0
#•	preprocess là pipeline chuẩn hóa ảnh (resize, normalize) để đưa vào model.
weights = EfficientNet_B0_Weights.DEFAULT
base_model = efficientnet_b0(weights=weights)
preprocess = weights.transforms()

# Lấy phần feature extractor
model = torch.nn.Sequential(
    base_model.features,
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten()
)
model.eval()

# Hàm chuyển ảnh thành vector
def image_to_vector(image_file):
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().cpu().numpy().tolist()

# 👉 Xử lý chuẩn hóa các trường dạng chuỗi thành list
#các trường ghi liền nhau như "vàng, xanh lá" -> "vàng", "xanh lá"
def normalize_payload_fields(payload):
    for key in ["color", "season"]:
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = [v.strip() for v in value.split(",")]
    return payload

# Chuẩn hóa kết quả
def normalize_result(item):
    if hasattr(item, 'payload') and hasattr(item, 'score'):
        payload = item.payload or {}
        score = item.score
    elif isinstance(item, dict):
        payload = item.get('payload', {})
        score = item.get('score')
    else:
        payload = {}
        score = None

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    payload = normalize_payload_fields(payload)

    return {
        'payload': {
            'name': payload.get('name', 'Không tên'),
            'image_url': payload.get('image_url', '/static/default.jpg'),
            'origin': payload.get('origin'),
            'color': payload.get('color'),
            'season': payload.get('season'),
            'category': payload.get('category'),
            'description': payload.get('description'),
            'keywords': payload.get('keywords'),
        },
        'score': round(score, 4)
    }

# Lấy danh sách filter options từ database
def get_filter_options():
    all_items = client.scroll(collection_name="fruit_text", with_payload=True, limit=500)[0]
    colors = set()
    seasons = set()
    origins = set()
    categories = set()
    
    for item in all_items:
        payload = item.payload
        if payload.get('color'):
            for c in str(payload['color']).split(','):
                colors.add(c.strip())
        if payload.get('season'):
            for s in str(payload['season']).split(','):
                seasons.add(s.strip())
        if payload.get('origin'):
            origins.add(payload['origin'].strip())
        if payload.get('category'):
            categories.add(payload['category'].strip())
    
    return {
        'colors': sorted(colors),
        'seasons': sorted(seasons),
        'origins': sorted(origins),
        'categories': sorted(categories)
    }

# Tìm kiếm mở rộng theo 4 thuộc tính
def search_by_attributes(collection, query):
    hits = client.scroll(
        collection_name=collection,
        with_payload=True,
        limit=400
    )[0]
    query_lower = query.strip().lower()
    matches = []
    #duyệt qua từng item
    for item in hits:
        payload = item.payload
        payload = normalize_payload_fields(payload)
        if any(query_lower in str(payload.get(field, '')).lower() for field in ['name', 'origin', 'color', 'season']):
            matches.append({
                'payload': payload,
                'score': 1.0
            })
    return matches

@app.route('/')
def home():
    banner = "banner1.jpg"
    return render_template('home.html', banner_url=banner)

# Tìm kiếm theo từ khóa
@app.route('/search-text', methods=['GET', 'POST'])
def search_text():
    results = []
    keyword = request.form.get('keyword', '').strip()
    color_filter = request.form.get('color', '')
    season_filter = request.form.get('season', '')
    origin_filter = request.form.get('origin', '')
    
    filter_options = get_filter_options()

    if request.method == 'POST':
        try:
            # Lấy tất cả items
            all_items = client.scroll(collection_name="fruit_text", with_payload=True, limit=500)[0]
            
            for item in all_items:
                payload = item.payload
                payload = normalize_payload_fields(payload)
                
                # Áp dụng filter
                if color_filter and color_filter.lower() not in str(payload.get('color', '')).lower():
                    continue
                if season_filter and season_filter.lower() not in str(payload.get('season', '')).lower():
                    continue
                if origin_filter and origin_filter.lower() not in str(payload.get('origin', '')).lower():
                    continue
                
                # Tìm theo keyword nếu có
                if keyword:
                    keyword_lower = keyword.lower()
                    if not any(keyword_lower in str(payload.get(field, '')).lower() 
                              for field in ['name', 'origin', 'color', 'season', 'description', 'keywords']):
                        continue
                
                results.append({'payload': payload, 'score': 1.0})
            
            # Nếu có keyword nhưng không tìm thấy bằng attribute, dùng semantic search
            if keyword and not results:
                query_vector = get_text_vector(keyword)
                raw_results = search_by_vector("fruit_text", query_vector, top_k=50)
                results = [normalize_result(item) for item in raw_results]

            if not results:
                flash("Không tìm thấy kết quả phù hợp.")
        except Exception as e:
            flash(f"Lỗi khi tìm kiếm: {e}")

    return render_template('search_text.html', results=results, keyword=keyword,
                          filter_options=filter_options, 
                          selected_color=color_filter,
                          selected_season=season_filter,
                          selected_origin=origin_filter)

# Tìm kiếm theo ảnh - trả về top 5 kết quả
@app.route('/search-image', methods=['GET', 'POST'])
def search_image():
    results = []

    if request.method == 'POST':
        image_file = request.files.get('image', None)
        if image_file and image_file.filename != '':
            try:
                vector = image_to_vector(image_file)
                raw_results = client.query_points(
                    collection_name="fruit_image",
                    query=vector,
                    limit=5,  # Trả về top 5 thay vì 1
                    with_payload=True,
                ).points
                results = [normalize_result(item) for item in raw_results]
                if not results:
                    flash("Không tìm thấy kết quả phù hợp cho ảnh đã chọn.")
            except Exception as e:
                flash(f'Có lỗi khi xử lý ảnh: {e}')
        else:
            flash("Vui lòng chọn một ảnh.")

    return render_template('search_image.html', results=results)

# Trang chi tiết trái cây
@app.route('/fruit/<name>')
def fruit_detail(name):
    try:
        all_items = client.scroll(
            collection_name="fruit_text",
            with_payload=True,
            limit=1000
        )[0]
        
        current_fruit = None
        current_vector = None
        
        for item in all_items:
            if item.payload.get('name', '').lower() == name.lower():
                item.payload = normalize_payload_fields(item.payload)
                current_fruit = item.payload
                # Lấy vector của trái cây hiện tại để tìm similar
                current_vector = get_text_vector(
                    (item.payload.get('description', '') + ' ' + item.payload.get('keywords', '')).strip()
                )
                break
        
        if not current_fruit:
            flash("Không tìm thấy thông tin chi tiết về trái cây.")
            return render_template("fruit_detail.html", fruit=None, similar_fruits=[])
        
        # Tìm trái cây tương tự
        similar_fruits = []
        if current_vector:
            similar_results = search_by_vector("fruit_text", current_vector, top_k=6)
            for item in similar_results:
                normalized = normalize_result(item)
                # Loại bỏ chính nó
                if normalized['payload']['name'].lower() != name.lower():
                    similar_fruits.append(normalized)
            similar_fruits = similar_fruits[:5]  # Lấy tối đa 5
        
        return render_template("fruit_detail.html", fruit=current_fruit, similar_fruits=similar_fruits)

    except Exception as e:
        flash(f"Lỗi khi tải thông tin chi tiết: {e}")
        return render_template("fruit_detail.html", fruit=None, similar_fruits=[])

# API so sánh 2 trái cây
@app.route('/compare', methods=['GET', 'POST'])
def compare_fruits():
    filter_options = get_filter_options()
    fruit1 = None
    fruit2 = None
    
    if request.method == 'POST':
        name1 = request.form.get('fruit1', '').strip()
        name2 = request.form.get('fruit2', '').strip()
        
        if name1 and name2:
            try:
                all_items = client.scroll(collection_name="fruit_text", with_payload=True, limit=500)[0]
                
                for item in all_items:
                    payload = normalize_payload_fields(item.payload.copy())
                    if payload.get('name', '').lower() == name1.lower():
                        fruit1 = payload
                    if payload.get('name', '').lower() == name2.lower():
                        fruit2 = payload
                    if fruit1 and fruit2:
                        break
                
                if not fruit1 or not fruit2:
                    flash("Không tìm thấy một hoặc cả hai loại trái cây.")
            except Exception as e:
                flash(f"Lỗi: {e}")
    
    # Lấy danh sách tên trái cây cho dropdown
    fruit_names = []
    try:
        all_items = client.scroll(collection_name="fruit_text", with_payload=True, limit=500)[0]
        fruit_names = sorted([item.payload.get('name', '') for item in all_items if item.payload.get('name')])
    except:
        pass
    
    return render_template('compare.html', fruit1=fruit1, fruit2=fruit2, fruit_names=fruit_names)

if __name__ == '__main__':
    app.run(debug=True)
