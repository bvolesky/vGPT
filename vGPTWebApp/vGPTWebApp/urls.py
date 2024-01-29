from django.urls import path, re_path, include
from chatbot import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat_view'),
    path('api/chatbot_response/', views.chatbot_response, name='chatbot_response'),
    re_path(r'^$', views.chat_view, name='root'),  # Redirect root URL to chat_view
]
