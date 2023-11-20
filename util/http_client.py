import requests
import json


def file_upload(url, files):
    r = requests.post(url, files=files)
    if 200 == r.status_code and r.text and json.loads(r.text)['code'] == 0:
        return json.loads(r.text)['data']
    else:
        return None


def data_post(url, data):
    r = requests.post(url, data=data)
    if 200 == r.status_code and r.text and json.loads(r.text)['code'] == 0:
        return 200
    else:
        return 500


def data_get(url, data):
    r = requests.get(url, data=data)
    if 200 == r.status_code and r.text and json.loads(r.text)['code'] == 0:
        return 200
    else:
        return 500
