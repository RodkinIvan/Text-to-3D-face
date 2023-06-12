import os
import sys
import requests
import time
import io
import zipfile
import json
import trimesh
import matplotlib.pyplot as plt

import gradio as gr

AUTH_FORM = {
    'grant_type': 'client_credentials',
    'client_id': 'XXX',
    'client_secret': 'XXX',
}

AUTH_FILE = 'oauth.json'
PLAYER_FILE = 'player.json'


def get_auth_header():
    now = time.time()
    access_token = None
    token_type = None

    if os.path.isfile(AUTH_FILE):
        with open(AUTH_FILE) as ifd:
            auth = json.load(ifd)
        expired = now > auth['expires']
        if not expired:
            access_token = auth['access_token']
            token_type = auth['token_type']
    request_new = not (access_token and token_type)
    if request_new:
        rsp = requests.post(
            'https://api.avatarsdk.com/o/token/',
            data=AUTH_FORM
        ).json()

        access_token = rsp['access_token']
        token_type = rsp['token_type']
        with open(AUTH_FILE, 'w') as ofd:
            rsp['expires'] = now + rsp['expires_in'] - 60
            json.dump(rsp, ofd)
    headers = {'Authorization': '{0} {1}'.format(token_type, access_token)}
    return headers


def get_player_uid_header(headers):
    player_uid = None

    if os.path.isfile(PLAYER_FILE):
        with open(PLAYER_FILE) as ifd:
            player = json.load(ifd)
        player_uid = player['code']

    if not player_uid:
        player_form = {'comment': 'test_py'}
        rsp = requests.post(
            'https://api.avatarsdk.com/players/',
            data=player_form, headers=headers
        ).json()

        player_id = rsp['code']

        with open(PLAYER_FILE, 'w') as ofd:
            json.dump(rsp, ofd)

    return {'X-PlayerUID': player_uid}


def convert_3d_file(image_file):
    headers = get_auth_header()

    headers.update(
        get_player_uid_header(headers)
    )

    pipeline_type = 'head_1.2'
    pipeline_subtype = 'base/mobile'

    rsp = requests.get(
        'https://api.avatarsdk.com/parameters/available/{}/'.format(pipeline_type),
        headers=headers,
        params={'pipeline_subtype': pipeline_subtype}
    ).json()

    available_resources = rsp[pipeline_subtype]
    resources = {}

    for category, category_value in available_resources.items():
        resources[category] = resources.get(category, {})
        for group, resource_names in category_value.items():
            resources[category][group] = [resource_names[0]]

    data = {
        'name': 'test',
        'pipeline': pipeline_type,
        'pipeline_subtype': pipeline_subtype,
        'resources': json.dumps(resources),
    }
    files = {'photo': open(image_file, 'rb')}
    rsp = requests.post(
        'https://api.avatarsdk.com/avatars/',
        headers=headers, data=data, files=files
    ).json()

    avatar_status_url = rsp['url']

    while True:
        rsp = requests.get(avatar_status_url, headers=headers).json()

        if rsp['status'] == 'Completed':
            print(rsp)
            break

        time.sleep(3)

    mesh = requests.get(rsp['mesh'], headers=headers)
    texture = requests.get(rsp['texture'], headers=headers)

    with io.BytesIO(mesh.content) as zipmemory:
        with zipfile.ZipFile(zipmemory) as archive:
            model_file = archive.namelist()[0]
            archive.extract(model_file)

    with open('model.jpg', 'wb') as texture_file:
        texture_file.write(texture.content)
    mesh = trimesh.load_mesh(model_file)
    mesh.show()

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print('Specify image file')
        sys.exit(1)

    convert_3d_file(sys.argv[1])