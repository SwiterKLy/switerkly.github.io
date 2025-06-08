from flask import Flask, redirect, request, session, url_for
import requests
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Для сессий

# Вставь сюда свои данные
CLIENT_KEY = "ТВОЙ_CLIENT_KEY"
CLIENT_SECRET = "ТВОЙ_CLIENT_SECRET"
REDIRECT_URI = "http://localhost:5000/callback"  # Зарегистрируй этот URL в настройках TikTok-приложения

SCOPES = "user.info.basic,video.upload,video.list,video.data.write,video.data.read"

@app.route("/")
def index():
    # Генерируем ссылку для авторизации
    auth_url = (
        f"https://open.tiktokapis.com/platform/oauth/connect/"
        f"?client_key={CLIENT_KEY}"
        f"&response_type=code"
        f"&scope={SCOPES}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&state=some_random_state"
    )
    return f'<a href="{auth_url}">Авторизоваться через TikTok</a>'

@app.route("/callback")
def callback():
    # TikTok редиректит сюда с ?code=...&state=...
    code = request.args.get("code")
    state = request.args.get("state")

    if not code:
        return "Ошибка: код авторизации не получен."

    # Обменяем код на access_token
    token_url = "https://open.tiktokapis.com/oauth/access_token/"
    data = {
        "client_key": CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI
    }
    resp = requests.post(token_url, json=data)
    if resp.status_code == 200:
        token_data = resp.json()
        access_token = token_data["data"]["access_token"]
        session["access_token"] = access_token
        return f"Access Token отримано: <br><pre>{access_token}</pre>"
    else:
        return f"Помилка отримання токена: {resp.text}"

if __name__ == "__main__":
    app.run(debug=True)
