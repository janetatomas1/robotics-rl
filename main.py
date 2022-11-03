
import docker
import json
from json.decoder import JSONDecodeError
import argparse
import pathlib
from datetime import datetime
import git
import string
import random


def main(settings):
    client = docker.DockerClient(version='auto')

    image_settings = settings["image"]
    image_name = image_settings.get("name", None)

    image = client.images.get(image_name)

    volume_settings = settings["volume"]
    volume_path = volume_settings.get("path")
    volume_host_path_prefix = volume_settings.get("host_path_prefix")
    container_name = ''.join(random.sample(string.ascii_lowercase + string.digits, 20))

    volume_host_path = "{}/{}".format(volume_host_path_prefix, container_name)

    internal_settings = json.dumps(settings.get("settings"))
    environment = {
        "VENV": "/opt/robotics-rl/venv/lib/python3.10/site-packages/",
        "settings": internal_settings,
    }

    client.containers.run(image=image.id, detach=True, name=container_name,
                          volumes={volume_host_path: {"bind": volume_path, "mode": "rw"}},
                          environment=environment)

    repository = git.Repo()
    settings["details"] = {
        "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        "commit_id": repository.head.object.hexsha,
        "branch": repository.active_branch.name,
    }

    with open("{}/settings.json".format(volume_host_path), "w") as json_file:
        json.dump(settings, fp=json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', default='settings/template.json',
                        help='JSON file containing pipeline settings value')

    args = parser.parse_args()

    settings_path = pathlib.Path(args.settings)
    if not settings_path.exists():
        raise Exception("settings file not found: {}".format(settings_path))

    try:
        settings_file = open(settings_path)
        settings = json.load(settings_file)
        settings_file.close()
    except JSONDecodeError as e:
        raise Exception("Unable to parse settings file: {}".format(settings_path))

    main(settings)
