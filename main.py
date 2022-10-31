
import docker
import json
from json.decoder import JSONDecodeError
import argparse
import pathlib


def main(settings):
    client = docker.DockerClient(version='auto')

    image_settings = settings["image"]
    build = image_settings.get("build", True)
    dockerfile = image_settings.get("path", None)
    image_name = image_settings.get("name", None)

    if build:
        image, _ = client.images.build(path=dockerfile, tag=image_name)
    else:
        image = client.images.get(name=image_name)

    container_settings = settings["container"]
    container_name = container_settings.get("name")

    try:
        container = client.containers.get(container_name)
        container.remove()
    except:
        pass

    volume_settings = settings["volume"]
    volume_path = volume_settings.get("path")
    volume_host_path_prefix = volume_settings.get("host_path_prefix")
    volume_host_path = "{}/{}".format(volume_host_path_prefix, container_name)

    algorithm_settings = settings.get("algorithm")
    rl_env_settings = settings.get("rl_env")
    policy_settings = settings.get("policy")

    environment = {
        "algorithm_settings": json.dumps(algorithm_settings),
        "rl_env_settings": json.dumps(rl_env_settings),
        "policy_settings": json.dumps(policy_settings),
        "VENV": "/opt/robotics-rl/venv/lib/python3.10/site-packages/",
    }

    client.containers.run(image=image_name, name=container_name, detach=True,
                          volumes={volume_host_path: {"bind": volume_path, "mode": "rw"}},
                          environment=environment)


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
