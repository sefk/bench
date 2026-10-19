from django.urls import path

from bench import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/rows/", views.api_rows, name="api_rows"),
    path("api/meta/", views.api_meta, name="api_meta"),
    path("api/new-mac/", views.api_new_mac, name="api_new_mac"),
]
