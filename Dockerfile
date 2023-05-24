# nini94/tensorflow2.5.0-dlib19.22-opencv4.5.2  download from dockerhup. you can also use our image aopdsenet.

FROM nini94/tensorflow2.5.0-dlib19.22-opencv4.5.2:latest

RUN useradd -ms /bin/bash cgpnk
USER cgpnk

WORKDIR /opt/cgpnk

ENV PATH="/home/cgpnk/.local/bin:${PATH}"

COPY --chown=cgpnk:cgpnk requirements.txt /opt/cgpnk/
COPY --chown=cgpnk:cgpnk model /opt/cgpnk/model/
COPY --chown=cgpnk:cgpnk multi_predict.py /opt/cgpnk/

RUN pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


ENTRYPOINT python3 -m multi_predict $0 $@s

