FROM ubuntu:22.04

RUN apt update -q && \
	export DEBIAN_FRONTEND=noninteractive && \
    apt install -y --no-install-recommends tar xz-utils \
        libx11-6 libxcb1 libxau6 libgl1-mesa-dev \
        xvfb dbus-x11 x11-utils libxkbcommon-x11-0 \
        libavcodec-dev libavformat-dev libswscale-dev \
        && \
    apt autoclean -y && apt autoremove -y && apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y software-properties-common
RUN apt update && apt install -y git vim wget curl python3-pip
RUN apt update && apt install -y libgl1-mesa-glx libffi-dev
RUN apt upgrade -y libstdc++6

WORKDIR /opt

RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu22_04.tar.xz

RUN tar -xf CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu22_04.tar.xz

ENV COPPELIASIM_ROOT=/opt/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu22_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

RUN mkdir robotics-rl
WORKDIR /opt/robotics-rl

RUN pip install virtualenv
RUN virtualenv venv

COPY requirements requirements

RUN . venv/bin/activate && pip install setuptools==66
RUN . venv/bin/activate && pip install -r requirements/dev.txt
RUN . venv/bin/activate && pip install git+https://github.com/stepjam/PyRep.git

COPY . .

ENV VENV=/opt/robotics-rl/venv/lib/python3.10/site-packages/

RUN apt update &&  apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
RUN apt update && nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# Install VirtualGL
RUN wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
RUN dpkg -i virtualgl*.deb

ENTRYPOINT ["bash", "scripts/run.sh"]
