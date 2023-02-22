FROM daisukekobayashi/darknet:cpu-cv

RUN apt update && apt install -y libsm6 libxext6 git
RUN apt-get install -y libxrender-dev python3 python3-pip

COPY src /opt/yolov4-evaluation/src
COPY Object-Detection-Metrics /opt/yolov4-evaluation/Object-Detection-Metrics

RUN cd /opt/yolov4-evaluation && pip3 install -r Object-Detection-Metrics/requirements.txt
