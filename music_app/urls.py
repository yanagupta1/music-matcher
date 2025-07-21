from django.urls import path
from .views import song_form, submit_songs

urlpatterns = [
    path('', song_form, name='song_form'),
    path('submit_songs/', submit_songs, name='submit_songs'),
]
