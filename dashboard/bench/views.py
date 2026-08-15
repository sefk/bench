from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render

from .loader import build_meta, load_rows


def index(request):
    return render(request, "bench/index.html")


def api_rows(request):
    rows = load_rows(settings.RESULTS_DIR)
    return JsonResponse({"rows": rows})


def api_meta(request):
    rows = load_rows(settings.RESULTS_DIR)
    return JsonResponse(build_meta(rows))
