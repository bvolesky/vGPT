from django.http import JsonResponse
from .app.scripts import toolkit
from django.shortcuts import render


def chat_view(request):
    return render(request, "chatbot/chat.html")


def chatbot_response(request):
    user_input = request.GET.get("user_input")
    response = toolkit.process_input(user_input)
    return JsonResponse({"response": response})
