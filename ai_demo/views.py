from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import whisper
import tempfile
import subprocess
import os

model = whisper.load_model("base", device="cpu")

def upload_page(request):
    return render(request, "upload.html")

@csrf_exempt
def transcribe_audio(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST request required"})

    media = request.FILES.get("media")
    if not media:
        return JsonResponse({"error": "No file uploaded"})

    # Save uploaded file (audio OR video)
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in media.chunks():
            temp.write(chunk)
        input_path = temp.name

    audio_path = input_path + ".wav"

    # 🔹 Always extract audio (works for audio + video)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        result = model.transcribe(audio_path, language="en")
        text = result.get("text", "").strip()

        if not text:
            return JsonResponse({"error": "No speech detected"})

        return JsonResponse({
            "text": text
        })

    except Exception as e:
        return JsonResponse({
            "error": "Audio processing failed",
            "details": str(e)
        })

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
