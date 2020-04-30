import requests
from typing import Dict

CACHE: Dict[str, requests.Response] = {}

def get(url, params, **kwargs):
    if url not in CACHE:
        CACHE[url] = requests.get(url, params, **kwargs)
    return CACHE[url]

