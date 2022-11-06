import docker
import string
import random
import os


def main():
    client = docker.DockerClient(version='auto')

    home_dir = os.environ["HOME"]
    volume_host_path_prefix = "{}/results".format(home_dir)
    container_name = ''.join(random.sample(string.ascii_lowercase + string.digits, 20))

    volume_host_path = "{}/{}".format(volume_host_path_prefix, container_name)

    environment = {
        "VENV": "/opt/robotics-rl/venv/lib/python3.10/site-packages/",
    }

    container = client.containers.run(image="thesis-image", detach=True, name=container_name,
                                      volumes={volume_host_path: {"bind": "/opt/results/", "mode": "rw"}},
                                      environment=environment)

    print(container.id)


if __name__ == "__main__":
    main()
