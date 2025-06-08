import requests

def upload_to_transfersh(file_path):
    file_name = file_path.split('/')[-1]
    with open(file_path, 'rb') as f:
        print(f"[INFO] Завантаження {file_name}...")
        response = requests.put(f'https://transfer.sh/{file_name}', data=f)
    
    if response.status_code == 200:
        print("[✅] Завантажено успішно!")
        print("🔗 Посилання:", response.text.strip())
    else:
        print("[❌] Помилка завантаження:", response.status_code)
        print(response.text)

if __name__ == "__main__":
    # Заміни шлях на своє зображення
    test_file = "static/images/Positive/Actor_0_7_Monkey_0_16.jpg"
    upload_to_transfersh(test_file)
