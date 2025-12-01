import os
import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

# Đặt backend 'Agg' cho matplotlib (non-interactive) để tránh lỗi thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify

# ======================
# ⚙️ Cấu hình Flask App
# ======================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ======================
# ⚙️ Thiết lập thiết bị
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Đang sử dụng:", device)

# ======================
# ⚙️ Đường dẫn mô hình
# ======================
model_path = "model/model_resnet18.pth"
classes_path = "model/classes.txt"

# ======================
# 📖 Đọc danh sách lớp
# ======================
with open(classes_path, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# ======================
# 🧠 Khởi tạo mô hình
# ======================
model = torchvision.models.resnet18(weights=None)
# Update the fully connected layer to match the training structure
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(classes))
)

# Load checkpoint đúng với cách bạn đã lưu
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])  # ✅ đúng với cấu trúc của bạn
model = model.to(device)
model.eval()

# ======================
# 🖼 Tiền xử lý ảnh
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # giữ nguyên như lúc train
])

# ======================
# 🔮 Hàm dự đoán
# ======================
def predict_image(image_path):
    try:
        # Kiểm tra file tồn tại
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy file: {image_path}")
            
        # Kiểm tra định dạng file có phải là ảnh
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"File không phải là ảnh hợp lệ: {str(e)}")
        
        # Xử lý ảnh và dự đoán
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            outputs = outputs / 1  # temperature scaling như bạn dùng
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            
        # Đảm bảo giải phóng bộ nhớ
        del tensor, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return probs
        
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path}: {str(e)}")
        raise

# ======================
# 📊 Hàm vẽ biểu đồ độ tin cậy
# ======================
def plot_confidence(probs):
    try:
        # Sắp xếp các lớp theo độ tin cậy giảm dần
        sorted_indices = probs.argsort()[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_classes = [classes[i] for i in sorted_indices]
        
        # Chỉ hiển thị top 8 lớp có độ tin cậy cao nhất
        top_n = min(8, len(sorted_classes))
        top_classes = sorted_classes[:top_n]
        top_probs = sorted_probs[:top_n]
        
        # Tạo bảng màu đẹp mắt
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_classes)))
        
        # Thiết lập kích thước và style
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        # Tạo figure với kích thước phù hợp
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        
        # Tạo thanh ngang với màu gradient đẹp mắt
        bars = ax.barh(top_classes, top_probs, color=colors, height=0.6, 
                      edgecolor='none', alpha=0.8)
        
        # Thêm nhãn phần trăm bên trong mỗi thanh
        for bar, prob in zip(bars, top_probs):
            width = bar.get_width()
            label_x_pos = width - 0.05 if width > 0.2 else width + 0.02
            label_alignment = 'right' if width > 0.2 else 'left'
            label_color = 'white' if width > 0.2 else 'black'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                   f"{prob*100:.1f}%", va='center', ha=label_alignment,
                   color=label_color, fontweight='bold', fontsize=10)
        
        # Thiết lập tiêu đề và nhãn
        ax.set_xlabel("Độ tin cậy", fontsize=12, labelpad=10)
        ax.set_xlim(0, 1.05)
        
        # Tùy chỉnh lưới và đường viền
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        
        # Đặt nhãn trục y sang trái và tăng kích thước
        ax.tick_params(axis='y', labelsize=11)
        ax.tick_params(axis='x', labelsize=10)
        
        # Thêm giá trị phần trăm trên trục x
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        plt.tight_layout()
        
        # Lưu vào buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close(fig)  # Đảm bảo đóng figure để giải phóng tài nguyên
        
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Lỗi khi tạo biểu đồ: {e}")
        # Trả về một hình ảnh trống nếu có lỗi
        empty_buf = io.BytesIO()
        fig, ax = plt.figure(figsize=(6, 2)), plt.axes()
        ax.text(0.5, 0.5, "Không thể tạo biểu đồ", ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(empty_buf, format='png', facecolor='white')
        empty_buf.seek(0)
        plt.close(fig)
        return base64.b64encode(empty_buf.getvalue()).decode("utf-8")

# ======================
# 🌐 Giao diện web
# ======================
def get_images():
    image_folder = "images"  # Thư mục images cùng cấp với app.py
    return [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra xem request có dữ liệu JSON không
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request phải có định dạng JSON'
            }), 400
            
        data = request.json
        image_path = data.get('image_path')
        
        # Kiểm tra image_path có tồn tại không
        if not image_path:
            return jsonify({
                'success': False,
                'error': 'Thiếu đường dẫn hình ảnh'
            }), 400
        
        # Trích xuất tên file từ đường dẫn
        if '/images/' in image_path:
            filename = image_path.split('/images/')[-1]
        else:
            filename = os.path.basename(image_path)
        
        # Đường dẫn đầy đủ đến file
        filepath = os.path.join('images', filename)
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return jsonify({
                'success': False,
                'error': f'Không tìm thấy file: {filename}'
            }), 404
        
        try:
            # Dự đoán
            probs = predict_image(filepath)
            top_idx = probs.argmax()
            result = classes[top_idx]
            confidence = probs[top_idx]
            
            # Nếu độ tin cậy dưới 70%, gán là "Unknown"
            if confidence < 0.7:
                result = "Unknown"
            
            # Tạo biểu đồ
            chart = plot_confidence(probs)
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': f"{confidence*100:.2f}%",
                'chart': chart
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Lỗi khi xử lý hình ảnh: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500
@app.route('/capture', methods=['POST'])
def capture():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            return jsonify({'success': False, 'error': 'Không có dữ liệu ảnh'}), 400

        # Giải mã base64 thành ảnh
        image_data = image_data.split(",")[1]  # loại bỏ prefix 'data:image/png;base64,'
        image_bytes = base64.b64decode(image_data)

        image_path = os.path.join(UPLOAD_FOLDER, "capture.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Dự đoán
        probs = predict_image(image_path)
        top_idx = probs.argmax()
        result = classes[top_idx]
        confidence = probs[top_idx]

        if confidence < 0.7:
            result = "Unknown"

        chart = plot_confidence(probs)

        return jsonify({
            'success': True,
            'result': result,
            'confidence': f"{confidence*100:.2f}%",
            'chart': chart
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi khi xử lý ảnh: {str(e)}'}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    images = get_images()
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Dự đoán
            probs = predict_image(filepath)
            top_idx = probs.argmax()
            result = classes[top_idx]
            confidence = probs[top_idx]

            # Nếu độ tin cậy dưới 70%, gán là "Unknown"
            if confidence < 0.7:
                result = "Unknown"

            chart = plot_confidence(probs)

            return render_template(
                "index.html",
                image_url=url_for("static", filename=f"uploads/{file.filename}"),
                result=result,
                confidence=f"{confidence*100:.2f}%",
                chart=chart,
                images=images
            )
    return render_template("index.html", image_url=None, images=images)

# ======================
# ▶️ Chạy web
# ======================
if __name__ == "__main__":
    app.run(debug=True)
