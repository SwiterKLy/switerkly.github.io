import os
import shutil
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from duckduckgo_search import DDGS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# --- Flask app
app = Flask(__name__)

# --- Шляхи до папок
POSITIVE_FOLDER = 'static/images/Positive'
NEGATIVE_FOLDER = 'static/images/Negative'
GENERATED_FOLDER = 'static/generated_images'
CUSTOM_FOLDER = 'saved/custom'

# --- Створення папок
os.makedirs(POSITIVE_FOLDER, exist_ok=True)
os.makedirs(NEGATIVE_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_FOLDER, exist_ok=True)

# --- Класи моделі
class EmbeddingNet(nn.Module):
    def __init__(self):
        
        super(EmbeddingNet, self).__init__()
        
        # Load a pre-trained ResNet50 model
        resnet = resnet50()
        
        # Freeze all layers except the last Convolution block
        for name, param in resnet.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
                
        # Define the embedding network by adding a few dense layers
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Exclude the last FC layer
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), SyncBatchNorm(512))
        self.dense2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), SyncBatchNorm(256))
        self.output = nn.Linear(256, 256)
        
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output(x)
        return x

class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor, positive, negative):
        ap_distance = F.pairwise_distance(anchor, positive, 2)
        an_distance = F.pairwise_distance(anchor, negative, 2)
        return ap_distance, an_distance

class TripletMarginLoss(nn.Module):
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, ap_distance, an_distance):
        return F.relu(ap_distance - an_distance + self.margin).mean()
    
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = embedding_net
    
    def forward(self, x1, x2):
        embed1 = self.embedding_net(x1)
        embed2 = self.embedding_net(x2)
        return embed1, embed2

# --- Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Whitelist для безпечного завантаження
torch.serialization.add_safe_globals({
    'SiameseNetwork': SiameseNetwork,
    'EmbeddingNet': EmbeddingNet,
    'DistanceLayer': DistanceLayer
})

# --- Завантаження моделі
model_path = 'Model.pth'
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# --- Трансформації
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0).to(device)

def calculate_similarity(img1: Image.Image, img2: Image.Image):
    tensor1 = preprocess_image(img1)
    tensor2 = preprocess_image(img2)
    with torch.no_grad():
        embed1, embed2 = model(tensor1, tensor2)
        dist = F.pairwise_distance(embed1, embed2).item()
        similarity = 1 / (1 + dist)
    return similarity

import time

def download_images(query, max_images):
    imgs = []
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for res in results:
            try:
                url = res['image']
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                imgs.append(img)
                time.sleep(1)  # ← додано паузу між завантаженнями
            except Exception:
                continue
    return imgs


@app.route('/')
def index():
    images = os.listdir(GENERATED_FOLDER)
    return render_template('index.html', images=images)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    target1 = data.get('target1')
    target2 = data.get('target2')
    quantity = int(data.get('quantity', 5))

    # Очистка папки
    for f in os.listdir(GENERATED_FOLDER):
        os.remove(os.path.join(GENERATED_FOLDER, f))

    imgs1 = download_images(target1, quantity)
    imgs2 = download_images(target2, quantity)

    threshold = 0.5
    saved_count = 0
    for i, img1 in enumerate(imgs1):
        for j, img2 in enumerate(imgs2):
            sim = calculate_similarity(img1, img2)
            if sim >= threshold:
                name1 = f"{target1}_{i}.jpg"
                path1 = os.path.join(GENERATED_FOLDER, name1)
                if not os.path.exists(path1):
                    img1.save(path1)
                    saved_count += 1
                name2 = f"{target2}_{j}.jpg"
                path2 = os.path.join(GENERATED_FOLDER, name2)
                if not os.path.exists(path2):
                    img2.save(path2)
                    saved_count += 1

    if saved_count == 0:
        return jsonify({'message': 'Не знайдено схожих зображень для генерації.'}), 200
    return jsonify({'message': f'Згенеровано {saved_count} зображень, схожих за запитами.'}), 200

@app.route('/sort', methods=['POST'])
def sort():
    data = request.get_json()
    selected_images = data.get('selected_images', [])
    folder_path = data.get('folder_path', 'Positive')

    if folder_path == 'Positive':
        target_folder = POSITIVE_FOLDER
    elif folder_path == 'Negative':
        target_folder = NEGATIVE_FOLDER
    else:
        target_folder = os.path.join(CUSTOM_FOLDER, folder_path)
        os.makedirs(target_folder, exist_ok=True)

    for img in selected_images:
        src = os.path.join(GENERATED_FOLDER, img)
        dst = os.path.join(target_folder, img)
        if os.path.exists(src):
            shutil.move(src, dst)

    for img in os.listdir(GENERATED_FOLDER):
        src = os.path.join(GENERATED_FOLDER, img)
        dst = os.path.join(NEGATIVE_FOLDER, img)
        shutil.move(src, dst)

    return jsonify({'message': f'Зображення переміщено до {folder_path}'}), 200

if __name__ == '__main__':
    app.run(debug=True)
