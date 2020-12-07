# dg_pytorch


## Installation

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

## Running the example

Generate the torch script module by running the python example

```
cd example
python pd.py
```

Then, compile the example.

```
cd example
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

To launch the compiled executeable, run from within the `example` directory

```
./build/pd script_pd_controller.pt
```




