
source: https://github.com/lllyasviel/ControlNet
        https://github.com/lllyasviel/ControlNet-v1-1-nightly

// runs locally with control
conda activate control

// how to run it locally -> WITH FLASK (compat old libs)
$env:FLASK_APP="../osais_ai_base/main_flask"
python -m flask run --host=0.0.0.0 --port=5108

// how to run it locally
uvicorn main:app --host 0.0.0.0 --port 5108

// how to build it 
// we prefer to use ./build.bat
docker build -t yeepeekoo/public:ai_cnet_ .  
docker build -t yeepeekoo/public:ai_cnet .  
docker push yeepeekoo/public:ai_cnet

// how to run it alongside a GATEWAY
docker run -d --name ai_cnet --gpus all --publish 5108:5108 --env-file env_docker_local yeepeekoo/public:ai_cnet

// how to run as VAI locally
1/ copy content of env_vai into env_local
2/ uvicorn main:app --host 0.0.0.0 --port 5108

// how to run it locally as a VAI in docker  (change REF to OSAIS in env file if needed)
docker run -d --name ai_cnet --gpus all --expose 5108 --publish 5108:5108 --env-file env_docker_local_vai yeepeekoo/public:ai_cnet
