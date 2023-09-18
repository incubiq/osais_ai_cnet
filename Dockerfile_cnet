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
COPY ./_temp/static/* ./static
COPY ./_temp/templates/* ./templates
COPY ./_temp/osais.json .
COPY ./_temp/main_fastapi.py .
COPY ./_temp/main_flask.py .
COPY ./_temp/main_common.py .

COPY ./_temp/osais_auth.py .
COPY ./_temp/osais_config.py .
COPY ./_temp/osais_inference.py .
COPY ./_temp/osais_main.py .
COPY ./_temp/osais_pricing.py .
COPY ./_temp/osais_s3.py .
COPY ./_temp/osais_training.py .
COPY ./_temp/osais_utils.py .

# keep the transparent image (case of using cnet with prompt only)
COPY ./_input/warmup.jpg ./_input/warmup.jpg
COPY ./_input/transparent.png ./_input/keep_transparent.png

# copy OSAIS mapping into AI
COPY ./cnet.json .

# overload config with those default settings
ENV ENGINE=cnet

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5108"]
