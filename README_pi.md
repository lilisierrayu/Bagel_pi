# Installation
```bash
uv venv --python 3.10 bagel
source bagel/bin/activate
uv pip install -r pi_requirements.txt
uv pip install -r requirements.txt
uv pip install numpy==1.26.4 sentencepiece==0.2.0 # requirements from pi_requirements.txt
uv pip install -e ../monopi
# run on GPU. if the download is unsuccessful, consider downgrade torch to 2.3.0 and upgrade back to 2.5.1 later
uv pip install flash_attn==2.5.8 --no-build-isolation 
```

# Train
```bash
sbatch train.sh <config>
```