import docker
import string
import random
import os


def main():
    client = docker.DockerClient(version='auto')

    home_dir = os.environ["HOME"]
    container_name = ''.join(random.sample(string.ascii_lowercase + string.digits, 20))

    results_path = "{}/results/{}".format(home_dir, container_name)

    container = client.containers.run(
        image="thesis-image",
        detach=True,
        name=container_name,
        volumes={
            results_path: {"bind": "/opt/results/", "mode": "rw"},
        }
    )
    print(container.id)


if __name__ == "__main__":
    main()
