import requests

def upload_to_0x0st(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('https://0x0.st', files=files)
    if response.status_code == 200:
        url = response.text.strip()
        print(f"✅ Файл успішно завантажено:\n{url}")
        return url
    else:
        print(f"❌ Помилка завантаження: {response.status_code}")
        return None

def post_to_tiktok(access_token, image_urls, title="funny cat", description="this will be a #funny photo on your @tiktok #fyp"):
    url = 'https://open.tiktokapis.com/v2/post/publish/content/init/'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }

    data = {
        "post_info": {
            "title": title,
            "description": description,
            "disable_comment": True,
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "auto_add_music": True
        },
        "source_info": {
            "source": "PULL_FROM_URL",
            "photo_cover_index": 1,
            "photo_images": image_urls
        },
        "post_mode": "DIRECT_POST",
        "media_type": "PHOTO"
    }

    response = requests.post(url, headers=headers, json=data)

    if response.ok:
        print("Публікація успішна:", response.json())
    else:
        print(f"Помилка публікації: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    access_token = "act.5tUGeDLpqFfBTu8UrSBkciFNVh5bYjXpLUjW6b8a4ALVvuC4V1LAAR923I8j!6356.va"
    file_path = "static/images/Positive/Actor_0_7_Monkey_0_16.jpg"

    uploaded_url = upload_to_0x0st(file_path)
    if uploaded_url:
        post_to_tiktok(access_token, [uploaded_url])
