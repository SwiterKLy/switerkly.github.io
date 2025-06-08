import os
import shutil
import time
from io import BytesIO

import requests
from PIL import Image
from flask import Flask, render_template, request, jsonify
from simple_image_download import simple_image_download as simp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from duckduckgo_search import DDGS
from torchvision.models import resnet50  # Импортировал, т.к. используешь в EmbeddingNet

# --- Константы путей
POSITIVE_FOLDER = 'static/images/Positive'
NEGATIVE_FOLDER = 'static/images/Negative'
GENERATED_FOLDER = 'static/generated_images'
CUSTOM_FOLDER = 'saved/custom'

# --- Создание папок, если их нет
os.makedirs(POSITIVE_FOLDER, exist_ok=True)
os.makedirs(NEGATIVE_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(CUSTOM_FOLDER, exist_ok=True)

# --- Инициализация Flask
app = Flask(__name__)


def is_black_image(image: Image.Image, threshold=5):
    """
    Проверяет, является ли изображение почти полностью черным.
    threshold — максимальное среднее значение (0–255), чтобы считать его черным.
    """
    grayscale = image.convert("L")  # переводим в ч/б
    mean_brightness = sum(grayscale.getdata()) / (grayscale.width * grayscale.height)
    return mean_brightness < threshold


# --- Определение моделей

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for name, param in resnet.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.SyncBatchNorm(512)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.SyncBatchNorm(256)
        )
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


# --- Устройство (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Безопасная загрузка модели (Whitelist)
torch.serialization.add_safe_globals({
    'SiameseNetwork': SiameseNetwork,
    'EmbeddingNet': EmbeddingNet,
    'DistanceLayer': DistanceLayer
})

# --- Загрузка модели
model_path = 'Model.pth'
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# --- Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Вспомогательные функции

def random_duckduckgo_filters():
    sizes = [None, "Small", "Medium", "Large", "Wallpaper"]
    colors = [None, "color", "monochrome", "red", "blue", "green"]
    layouts = [None, "Square", "Tall", "Wide"]
    types = [None, "photo", "clipart", "gif"]

    return {
        "size": random.choice(sizes),
        "color": random.choice(colors),
        "layout": random.choice(layouts),
        "type_image": random.choice(types),
    }


def preprocess_image(image: Image.Image):
    """Применить трансформации и подготовить тензор"""
    return transform(image).unsqueeze(0).to(device)

def calculate_similarity(img1: Image.Image, img2: Image.Image):
    """Вычислить сходство между двумя изображениями через модель"""
    tensor1 = preprocess_image(img1)
    tensor2 = preprocess_image(img2)
    with torch.no_grad():
        embed1, embed2 = model(tensor1, tensor2)
        dist = F.pairwise_distance(embed1, embed2).item()
        similarity = 1 / (1 + dist)
    return similarity

def download_images(query, max_images):
    """Завантажити зображення за допомогою simple_image_download"""
    response = simp.simple_image_download()
    temp_dir = "temp_download"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        response.download(query, limit=max_images)
        query_folder = os.path.join("simple_images", query.replace(" ", "_"))
        image_files = os.listdir(query_folder)
        imgs = []

        for file in image_files:
            try:
                img_path = os.path.join(query_folder, file)
                img = Image.open(img_path).convert('RGB')
                if not is_black_image(img):
                    imgs.append(img)
            except Exception:
                continue

        # Очистити тимчасову папку після використання
        shutil.rmtree("simple_images", ignore_errors=True)
        return imgs
    except Exception as e:
        print(f"Error downloading images: {e}")
        return []


# --- Flask маршруты

@app.route('/')
def index():
    images = os.listdir(GENERATED_FOLDER)
    return render_template('index.html', images=images)

def combine_images(img1: Image.Image, img2: Image.Image, final_size=(400, 200)) -> Image.Image:
    """
    Создать изображение final_size, поделённое пополам по ширине:
    левая половина - img1, правая - img2.
    Каждый из входных образов масштабируется в (final_width/2, final_height).
    """
    final_width, final_height = final_size
    half_width = final_width // 2

    # Масштабируем изображения к размеру половины итогового изображения
    img1_resized = img1.resize((half_width, final_height), Image.Resampling.LANCZOS)
    img2_resized = img2.resize((half_width, final_height), Image.Resampling.LANCZOS)

    # Создаём новое изображение
    combined = Image.new('RGB', final_size)
    combined.paste(img1_resized, (0, 0))
    combined.paste(img2_resized, (half_width, 0))

    return combined
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    target1 = data.get('target1')
    target2 = data.get('target2')
    required_quantity = int(data.get('quantity', 5))

    # Очистити папку GENERATED_FOLDER
    for f in os.listdir(GENERATED_FOLDER):
        os.remove(os.path.join(GENERATED_FOLDER, f))

    threshold = 0.8
    saved_count = 0
    batch_size = 40  # Скільки зображень завантажувати за одну ітерацію
    attempt = 0
    max_attempts = 10  # Захист від нескінченного циклу test

    used_pairs = set()

    while saved_count < required_quantity and attempt < max_attempts:
        imgs1 = download_images(target1, batch_size)
        imgs2 = download_images(target2, batch_size)

        for i, img1 in enumerate(imgs1):
            for j, img2 in enumerate(imgs2):
                if (i, j) in used_pairs:
                    continue

                sim = calculate_similarity(img1, img2)
                if sim >= threshold:
                    combined_img = combine_images(img1, img2, final_size=(400, 200))
                    combined_name = f"{target1}_{attempt}_{i}_{target2}_{attempt}_{j}.jpg"
                    combined_path = os.path.join(GENERATED_FOLDER, combined_name)
                    combined_img.save(combined_path)
                    saved_count += 1
                    if saved_count >= required_quantity:
                        break
                used_pairs.add((i, j))
            if saved_count >= required_quantity:
                break

        attempt += 1

    if saved_count == 0:
        return jsonify({'message': 'Не знайдено жодного схожого зображення для генерації.'}), 200
    elif saved_count < required_quantity:
        return jsonify({'message': f'Згенеровано тільки {saved_count} з {required_quantity} бажаних зображень.'}), 200
    else:
        return jsonify({'message': f'Успішно згенеровано {saved_count} комбінованих зображень.'}), 200


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

    # Переместить выбранные изображения в целевую папку
    for img in selected_images:
        src = os.path.join(GENERATED_FOLDER, img)
        dst = os.path.join(target_folder, img)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Остальные изображения из GENERATED_FOLDER в Negative
    for img in os.listdir(GENERATED_FOLDER):
        src = os.path.join(GENERATED_FOLDER, img)
        dst = os.path.join(NEGATIVE_FOLDER, img)
        shutil.move(src, dst)

    return jsonify({'message': f'Зображення переміщено до {folder_path}'}), 200

# --- Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
