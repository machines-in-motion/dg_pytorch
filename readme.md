# dg_pytorch

### What it is

A dynamic graph entity wrapper for running pytorch/torch script modules.

### Installation

Install the pytorch python library:

```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

In addition, we need the C++ library for using pytorch. This library is provided by pytorch. However, when doing experiments this lead to `MemoryErrors` and `bad_alloc` errors.

A workaround is to build the library from source. This can be done as follows (based on the installation guide from [this page](https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst)):

```
cd ~Downloads/
git clone -b master --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir pytorch-build pytorch-install
cd pytorch-build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch -- -j16
cmake --build . --target install
```

which builds the library using 16 threads (note the `-j16` at the end of the line).

After building the library, install it to `/opt/libtorch`:

```
cd ~Downloads/
sudo mv pytorch-install /opt/libtorch
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

### Basic API

A `dg_pytorch` entity corresponds to one script module (neural network model).


To demonstrate the API, let's assume we generated a script module that implements a PD controller with P and D gains as follows:

```python
class PDController(torch.nn.Module):
    def __init__(self):
        super(PDController, self).__init__()

    def forward(self, kp, kd, position, velocity, des_position, des_velocity):
        return kp * (des_position - position) + kd * (des_velocity - velocity)

pd_controller = PDController()
sm = torch.jit.script(pd_controller)

sm.save("script_pd_controller.pt")
```


You can load your previously generated torch script model by calling

```
pt_entity.load_model("script_pd_controller.pt")
```

Make sure you have the correct working directy when loading the script from dynamic graph.

Once you loaded the model, you have to sepcify the input and output signals in dynamic graph. This is done using the `add_input` and `add_output` function. Note that the ordering of the `add_input` corresponds to the ordering of the arguments in the `forward` method. For now we only support pytorch tensors as arguments. There can be multiple output tensors (by returning a tuple from the `forward` method.

Here are the input and output definitions for the above example:

```python
# Add the inputs.
pt_entity.add_input("sin_kp")
pt_entity.add_input("sin_kd")
pt_entity.add_input("sin_position")
pt_entity.add_input("sin_velocity")
pt_entity.add_input("sin_des_position")
pt_entity.add_input("sin_des_velocity")

# Define the output
pt_entity.add_output('sout_torque')
```

Note that the defined signals are not available as properties on the entity (e.g. `pt_entity.sin_kp` won't work). Instead, you have to use the `signal` method to access the signals like `pt_entity.signal('sin_kp')`.

Also note that the API is working tensors/arrays. Therefore, even when you are setting only a single scalar like for the above `kp`, you are supposed to define the value using `np.array()`, like

```
pt_entity.signal('sin_kp').value = np.array([5.])
```

#### Warmup and timing

Evaluating the network the first few times takes significantly more time than afterwards. Especially, the first few times might exceed the timing budget for the realtime computation (more than 1 ms). To avoid this, there is a special

```python
pt_entity.warmup()
```

method. This runs the network three times given the provided input. The idea is that all input signals should be set with realistic dummy values before calling `warmup()` to make sure internally vectors of the right size are allocated.

You can get the duration for executing the model by looking at the `last_run_duration_ms` signal:

```
pt_entity.last_run_duration_ms.value
```

This signal provides the duration it took to run the network the last time in milliseconds.


### Authors

- Julian Viereck

### Copyrights

Copyright(c) 2020 Max Planck Gesellschaft, New York University

### License

BSD 3-Clause License


