
command="python3 src/main.py"

echo xvfb-run --server-args "-ac -screen 0, 1024x1024x24" ${command} "$@"

xvfb-run --server-args "-ac -screen 0, 1024x1024x24" ${command} "$@"
