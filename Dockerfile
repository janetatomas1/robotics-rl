FROM ubuntu:22.04

RUN apt update && apt install -y software-properties-common
RUN apt update && apt install -y git vim wget curl python3-pip
RUN apt update && apt install -y libgl1-mesa-glx libffi-dev

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

RUN . venv/bin/activate && pip install -r requirements/dev.txt
RUN . venv/bin/activate && pip install git+https://github.com/stepjam/PyRep.git

COPY . .

RUN apt upgrade -y libstdc++6
