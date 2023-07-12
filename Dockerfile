##
##      To build the AI_CNET docker image
##

# base stuff
FROM yeepeekoo/public:ai_cnet_

## keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai
COPY ./ai/models/v1-5-pruned.ckpt ./ai/models/v1-5-pruned.ckpt
COPY ./ai/models/control_v11p_sd15_canny.yaml ./ai/models/control_v11p_sd15_canny.yaml
COPY ./ai/models/control_v11p_sd15_canny.pth ./ai/models/control_v11p_sd15_canny.pth
COPY ./ai/models/clip-vit-large-patch14 ./ai/models/clip-vit-large-patch14
COPY ./ai/annotator ./ai/annotator
COPY ./ai/cldm ./ai/cldm
COPY ./ai/ldm ./ai/ldm
COPY ./ai/share.py ./ai/share.py
COPY ./ai/config.py ./ai/config.py
COPY ./ai/runai.py ./ai/runai.py

# push again the base files
COPY ./_temp/main_fastapi.py .
COPY ./_temp/main_flask.py .
COPY ./_temp/main_common.py .
COPY ./_temp/osais_debug.py .
COPY ./_temp/osais.json .

# copy OSAIS mapping into AI
COPY ./cnet.json .
COPY ./_cnet.py .

# overload config with those default settings
ENV ENGINE=cnet

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5108"]
