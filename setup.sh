pip install -r requirements.txt
python setup.py develop
conda install pytorch=1.5.0 torchvision cudatoolkit=10.0 -c pytorch
conda install faiss-gpu cudatoolkit=10.0 -c pytorch

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./