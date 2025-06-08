import subprocess
import requests
import time
import os
import traceback

def create_video_with_transitions(
    image_files, 
    output_file='intermediate.mp4', 
    width=500, 
    height=500, 
    duration_per_image=3, 
    transition_duration=1
):
    try:
        inputs = []
        filter_parts = []
        
        for i, img in enumerate(image_files):
            inputs += ['-loop', '1', '-t', str(duration_per_image), '-i', img]
            filter_parts.append(
                f'[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];'
            )
        
        if len(image_files) < 2:
            filter_complex = ''.join(filter_parts)
            last_label = f'v0'
        else:
            filter_parts.append(
                f'[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={duration_per_image - transition_duration}[x1];'
            )
            for i in range(2, len(image_files)):
                offset = (duration_per_image - transition_duration) * (i - 1) + (duration_per_image - transition_duration)
                filter_parts.append(
                    f'[x{i-1}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}[x{i}];'
                )
            filter_complex = ''.join(filter_parts)
            last_label = f'x{len(image_files)-1}'
        
        cmd_video = ['ffmpeg', '-y', *inputs,
                     '-filter_complex', filter_complex,
                     '-map', f'[{last_label}]',
                     '-pix_fmt', 'yuv420p', '-c:v', 'libx264', output_file]
        
        print("Создаем видео с переходами...")
        subprocess.run(cmd_video, check=True)
        print("Видео создано:", output_file)
    except subprocess.CalledProcessError as e:
        print("Ошибка при создании видео с переходами:")
        print(e)
        traceback.print_exc()
        raise
    except Exception as e:
        print("Неожиданная ошибка при создании видео:")
        print(e)
        traceback.print_exc()
        raise


def add_audio_to_video(video_file, audio_file, output_file='output.mp4'):
    try:
        cmd_audio = ['ffmpeg', '-y', '-i', video_file, '-i', audio_file,
                     '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_file]
        print("Добавляем аудио...")
        subprocess.run(cmd_audio, check=True)
        print("Аудио добавлено, итоговое видео:", output_file)
    except subprocess.CalledProcessError as e:
        print("Ошибка при добавлении аудио:")
        print(e)
        traceback.print_exc()
        raise
    except Exception as e:
        print("Неожиданная ошибка при добавлении аудио:")
        print(e)
        traceback.print_exc()
        raise


def get_access_token(flask_server_url):
    for i in range(30):
        try:
            resp = requests.get(f"{flask_server_url}/token")
            data = resp.json()
            if "access_token" in data:
                token = data["access_token"]
                print("Access token получен")
                return token
            else:
                print(f"Токен еще не получен, попытка {i+1}/30...")
        except Exception as e:
            print(f"Ошибка при запросе токена, попытка {i+1}/30:", e)
            traceback.print_exc()
        time.sleep(1)
    raise Exception("Не удалось получить токен за 30 секунд")


def init_upload(access_token, video_size):
    try:
        url = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        if response.status_code == 200 and data.get("error", {}).get("code") == "ok":
            print("Инициализация загрузки прошла успешно")
            return data["data"]["upload_url"], data["data"]["publish_id"]
        else:
            print("Ошибка инициализации загрузки. Ответ сервера TikTok:", response.status_code, response.text)
            return None, None
    except Exception as e:
        print("Ошибка при инициализации загрузки:")
        print(e)
        traceback.print_exc()
        raise


def upload_video_chunk(upload_url, video_path):
    try:
        video_size = os.path.getsize(video_path)
        with open(video_path, "rb") as f:
            video_data = f.read()

        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Range": f"bytes 0-{video_size-1}/{video_size}"
        }
        response = requests.put(upload_url, headers=headers, data=video_data)
        if response.status_code in [200, 201]:
            print("Видео успешно загружено")
            return True
        else:
            print(f"Ошибка загрузки видео: статус {response.status_code}, ответ: {response.text}")
            return False
    except Exception as e:
        print("Ошибка при загрузке видео:")
        print(e)
        traceback.print_exc()
        raise


def publish_video(access_token, publish_id, video_size, text="Видео через API"):
    try:
        url = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "publish_id": publish_id,
            "text": text,
            "visibility_type": "PUBLIC",
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_size,
                "chunk_size": video_size,
                "total_chunk_count": 1
            }
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.ok:
            print("Видео опубликовано успешно!")
            return True
        else:
            print(f"Ошибка публикации видео: статус {response.status_code}, ответ: {response.text}")
            return False
    except Exception as e:
        print("Ошибка при публикации видео:")
        print(e)
        traceback.print_exc()
        raise


def create_and_upload_video(
    image_files,
    audio_file,
    tiktok_text,
    flask_server_url="https://switerkly.pythonanywhere.com",
    intermediate_video='intermediate.mp4',
    final_video='output.mp4'
):
    try:
        # Создаем видео с переходами из изображений
        create_video_with_transitions(image_files, intermediate_video)
        
        # Добавляем аудио
        add_audio_to_video(intermediate_video, audio_file, final_video)
        
        # Получаем токен
        token = get_access_token(flask_server_url)
        
        # Инициализируем загрузку
        video_size = os.path.getsize(final_video)
        upload_url, publish_id = init_upload(token, video_size)
        if not upload_url or not publish_id:
            raise RuntimeError("Ошибка инициализации загрузки")
        
        # Загружаем видео
        if not upload_video_chunk(upload_url, final_video):
            raise RuntimeError("Ошибка загрузки видео")
        
        # Публикуем видео
        if not publish_video(token, publish_id, video_size, tiktok_text):
            raise RuntimeError("Ошибка публикации видео")
        
        print("Видео успешно опубликовано!")
        return publish_id  # Возвращаем ID публикации для проверки статуса
    except Exception as e:
        print("Ошибка в процессе создания и загрузки видео:")
        print(e)
        traceback.print_exc()
        raise
