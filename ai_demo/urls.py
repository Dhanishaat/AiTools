from django.urls import path
from .views import upload_page, transcribe_audio

urlpatterns = [
    path("", upload_page, name="upload"),
    path("transcribe/", transcribe_audio, name="transcribe"),
]
