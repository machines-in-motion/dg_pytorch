# dg_pytorch

### What it is

A dynamic graph entity wrapper for running pytorch/torch script modules.

### Installation

Install pytorch:

```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Download the libtorch distribution and place it somewhere. Here we assume the files got moved under /opt/

```
cd ~/Downloads/
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.7.0+cpu.zip
sudo mv libtorch /opt/
```

### Running the example

Generate the torch script module by running the python example

```
cd demo
python pd.py
```

After compiling the repo, you should be able to run the entity demo by running

```
python entity_pd.py
```

### Authors

- Julian Viereck

### Copyrights

Copyright(c) 2020 Max Planck Gesellschaft, New York University

### License

BSD 3-Clause License


