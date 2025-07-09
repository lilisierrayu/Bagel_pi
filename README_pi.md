# Installation
```bash
uv venv --python 3.11.9
source .venv/bin/activate
uv pip sync pi_requirements.txt
uv pip sync requirements.txt
uv pip install numpy==1.26.4 sentencepiece==0.2.0 # requirements in pi_requirements.txt
uv pip install flash_attn==2.5.8 --no-build-isolation
```

# Train
```bash
./train.sh
```