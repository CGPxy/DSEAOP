FROM ubuntu:v1

USER cgpnk

WORKDIR /home/dy/DSENET

ENV PATH="/anaconda3/bin:$PATH"

COPY --chown=cgpnk:cgpnk requirements.txt /home/DSENET/
COPY --chown=cgpnk:cgpnk model/ /home/DSENET/model/
COPY --chown=cgpnk:cgpnk multi_predict.py /home/DSENET/

ENTRYPOINT python -m multi_predict $0 $@

LABEL nl.diagnijmegen.rse.algorithm.name=seg_algorithm


