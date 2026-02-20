from django.urls import path
from .views import upload_page, transcribe_audio, lecture_chatbot, summarize_text

urlpatterns = [
    path("", upload_page, name="upload"),
    path("transcribe/", transcribe_audio, name="transcribe"),
    path("summarize/", summarize_text),   # ✅ REQUIRED

    path("chat/", lecture_chatbot),   # 👈 chatbot

]
