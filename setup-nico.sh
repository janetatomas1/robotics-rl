
#!/bin/bash

virtualenv --python=python3.7 ~/.NICO-python3

source ~/.NICO-python3/bin/activate

pip install "setuptools==57.5.0"

source NICO-software/api/NICO-python3.bash

pip install -r requirements/dev.txt
