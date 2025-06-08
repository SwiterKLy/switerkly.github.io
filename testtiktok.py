import requests
import os

ACCESS_TOKEN = "P3Ce3DaY2yeUGccDvK3U1Ev2nHqxvRa1WN1k-x1UlfDldqAWeKmrC9GyaNL5qBhxsmb6iKicFgSVYHjwb8rJRNaD2Ug4JP3Px2mQxbZ9hFgXVwaitCipiJvkzc2rW1XmSzcsD1shW3KRSpUTllC-dpzagMA_xCn6X2AqJ44K8ytwu-wmFy7l4O6j7uNdXmcq*0!6369.va"
API_BASE = "https://open.tiktokapis.com"
VIDEO_FILE_PATH = "/home/switerkly/Загрузки/supawork-c4d0dc52728947e4839afa8d8c54ce59(1)(1).mp4"

def init_video_upload(file_size):
    url = f"{API_BASE}/v2/post/publish/video/init/"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    # Если файл меньше 10 МБ, chunk_size = размер файла
    chunk_size = file_size if file_size < 10_000_000 else 10_000_000
    total_chunks = (file_size + chunk_size - 1) // chunk_size
    json_data = {
        "post_info": {
            "title": "Test video upload via API",
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_comment": True,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": chunk_size,
            "total_chunk_count": total_chunks
        }
    }
    print("Отправляем init видео запрос с телом:")
    print(json_data)
    resp = requests.post(url, headers=headers, json=json_data)
    print("Ответ от сервера:", resp.status_code, resp.text)
    resp.raise_for_status()
    data = resp.json()
    upload_url = data["data"]["upload_url"]
    publish_id = data["data"]["publish_id"]
    return upload_url, publish_id, chunk_size, total_chunks



def upload_video_chunks(upload_url, file_path, chunk_size, total_chunks):
    with open(file_path, "rb") as f:
        for chunk_index in range(total_chunks):
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Range": f"bytes {chunk_index*chunk_size}-{chunk_index*chunk_size + len(chunk_data) - 1}/*"
            }
            resp = requests.put(upload_url, headers=headers, data=chunk_data)
            resp.raise_for_status()
            print(f"Chunk {chunk_index + 1}/{total_chunks} uploaded")

def complete_upload(publish_id):
    url = f"{API_BASE}/v2/post/publish/video/complete/"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    json_data = {
        "publish_id": publish_id
    }
    resp = requests.post(url, headers=headers, json=json_data)
    resp.raise_for_status()
    print("Upload completed and video published.")
    print(resp.json())

def main():
    file_size = os.path.getsize(VIDEO_FILE_PATH)
    upload_url, publish_id, chunk_size, total_chunks = init_video_upload(file_size)
    upload_video_chunks(upload_url, VIDEO_FILE_PATH, chunk_size, total_chunks)
    complete_upload(publish_id)

if __name__ == "__main__":
    main()
