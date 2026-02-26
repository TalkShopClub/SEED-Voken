conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n seed-voken python=3.10.12 -y && conda run -n seed-voken pip install -r requirements.txt
conda init