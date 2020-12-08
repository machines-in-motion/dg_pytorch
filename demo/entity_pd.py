import numpy as np
import dg_pytorch.dynamic_graph.dg_pytorch_entities as dg_pytorch_entities

# Create a pytorch entity.
pt_entity = dg_pytorch_entities.PyTorchEntity('PyTorchTestEntity')

# Raise error when the script file is not available
try:
    pt_entity.load_model("FooBar")
except RuntimeError:
    print("Got expected runtime error")

pt_entity.load_model("script_pd_controller.pt")

# Add the inputs.
pt_entity.add_input("sin_kp")
pt_entity.add_input("sin_kd")
pt_entity.add_input("sin_position")
pt_entity.add_input("sin_velocity")
pt_entity.add_input("sin_des_position")
pt_entity.add_input("sin_des_velocity")

# Define the output
pt_entity.add_output('sout_torque')

# Set some intial values.
pt_entity.signal('sin_kp').value = np.array([5.])
pt_entity.signal('sin_kd').value = np.array([0.01])
pt_entity.signal('sin_des_position').value = np.array([1., 2., 3., 4., 5., 6.])
pt_entity.signal('sin_position').value = np.array([0., 1., 2., 3., 4., 4.])

pt_entity.signal('sin_des_velocity').value = np.array([0., 0., 0., 0., 0., 0.])
pt_entity.signal('sin_velocity').value = np.array([0., 0., 0., 1., 0., 0.])

# Warmup the network.
pt_entity.warmup()

# Query the outputs
for t in range(5):
    pt_entity.signal('sout_torque').recompute(t)
    print(pt_entity.signal('sout_torque').value)
