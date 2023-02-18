
docker build -f Dockerfile.nico . -t thesis-image

source venv/bin/activate

python main.py
