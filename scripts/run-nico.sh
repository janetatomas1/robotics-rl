
source /root/.NICO-python3/bin/activate

command="python3 -m src.main"

xvfb-run --server-args "-ac -screen 0, 1024x1024x24" ${command} "$@"
