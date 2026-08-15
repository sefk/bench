import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-only-not-secret"

DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# Directory that holds results/<date>/<slug>.json files. Defaults to the
# sibling `results/` directory at the repo root, one level up from
# `dashboard/`. Override with the BENCH_RESULTS_DIR env var.
RESULTS_DIR = Path(
    os.environ.get("BENCH_RESULTS_DIR", str(BASE_DIR.parent / "results"))
)

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "bench",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# No database is used by this project.
DATABASES = {}

USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
