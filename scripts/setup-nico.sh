
#!/bin/bash

virtualenv --python=python3.7 ~/.NICO-python3

/root/.NICO-python3/bin/pip install "setuptools==57.5.0"

source NICO-software/api/NICO-python3.bash

/root/.NICO-python3/bin/pip install -r dev.txt
