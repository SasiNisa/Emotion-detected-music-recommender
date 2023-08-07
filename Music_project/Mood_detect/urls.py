from django.urls import path
from . import views

urlpatterns= [
    path("", views.index, name="home"),
    #path("recommend_music",views.recommend_music),
    
]