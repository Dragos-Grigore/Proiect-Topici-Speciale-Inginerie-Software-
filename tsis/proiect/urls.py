from django.urls import path
from . import views

urlpatterns = [
    path('run/', views.multimodal_page, name='multimodal_page'),
    path('upload/', views.upload_image, name='upload_image')
]