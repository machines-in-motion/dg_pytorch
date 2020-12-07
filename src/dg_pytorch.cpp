/**
 * @file dg_pytorch.hpp
 * @author Julian Viereck
 * @license License BSD-3-Clause
 * @copyright Copyright (c) 2019, New York University and Max Planck Gesellschaft.
 * @date 2020-12-07
 * @brief Pytorch bindings for dynamic graph.
 */

#include <dynamic-graph/factory.h>
#include <dynamic-graph/all-commands.h>

#include<dg_pytorch/dg_pytorch.hpp>

using namespace std;
using namespace dynamicgraph;


namespace dg_pytorch {

DYNAMICGRAPH_FACTORY_ENTITY_PLUGIN(PyTorchEntity, "PyTorchEntity");

PyTorchEntity::PyTorchEntity( const std::string& name ):
  Entity(name),
  // Define a signal that is always ready so the output signal is evaluated
  internal_signal_refresher_("PyTorchEntity("+name+")::intern(dummy)::refresher" ),
  signal_run_network_(
      boost::bind(&PyTorchEntity::run_network, this, _1, _2),
      internal_signal_refresher_,
      "PyTorchEntity("+name+")::output(double)::last_run_duration_ms"
  )
{
  // Define the refresh signal as always ready.
  internal_signal_refresher_.setDependencyType(
    dynamicgraph::TimeDependency<int>::ALWAYS_READY);

  // Make the timing signal available on the entity.
  signalRegistration(signal_run_network_);

  //  Add commands:
  addCommand (
    "load_model",
    dynamicgraph::command::makeCommandVoid1(
      *this,
      &PyTorchEntity::load_model,
      dynamicgraph::command::docCommandVoid1(
          "Loads a saved script model",
          "string: path to saved model")
    )
  );

  addCommand (
    "warmup",
    dynamicgraph::command::makeCommandVoid0(
      *this,
      &PyTorchEntity::warmup,
      dynamicgraph::command::docCommandVoid0(
          "Warmup the model by evaluating it three times")
    )
  );

  addCommand (
    "add_input",
    dynamicgraph::command::makeCommandVoid1(
      *this,
      &PyTorchEntity::add_input,
      dynamicgraph::command::docCommandVoid1(
          "Add input signal for input tensor.",
          "string: signal name")
    )
  );

  addCommand (
    "add_output",
    dynamicgraph::command::makeCommandVoid1(
      *this,
      &PyTorchEntity::add_output,
      dynamicgraph::command::docCommandVoid1(
          "Add output signal for output tensor.",
          "string: signal name")
    )
  );
}

void PyTorchEntity::load_model(const std::string& script_module)
{
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(script_module);
  }
  catch (const c10::Error& e) {
    throw std::runtime_error("PyTorchEntity: Unable to load script module at " + script_module);
  }
}

void PyTorchEntity::add_input(const std::string& signal_name)
{
    // Create the input signal.
    std::unique_ptr<SignalIn> signal;
    signal.reset(new SignalIn (
        NULL, getClassName() + "(" + getName() + ")::input(vector)::" + signal_name
    ));
    signalRegistration(*signal);

    // Add dependency for running the network to the new input signal.
    signal_run_network_.addDependency(*signal);

    // Store the mapping between signal and operator name
    input_signals_.push_back(std::make_pair(dg::Vector(), std::move(signal)));

    // Resize the input vector to avoid allocations later on.
    net_inputs_.resize(input_signals_.size());
}

void PyTorchEntity::add_output(const std::string& signal_name)
{
  // Create the output signal.
  std::unique_ptr<SignalOut> signal;
  signal.reset(new SignalOut(
      boost::bind(&PyTorchEntity::signal_callbacks, this, signal_name, _1, _2),
      signal_run_network_,
      getClassName() + "(" + getName() + ")::output(vector)::" + signal_name
  ));
  signalRegistration(*signal);

  // Store the output signal to avoid it getting deallocated.
  output_signals_.push_back(std::move(signal));
  output_signal_names_.push_back(signal_name);
}

double& PyTorchEntity::run_network(double& res, const int& time)
{
  static int i = 0;
  run_timer_.tic();

  // Iterate over all input tensors and copy their values to the network.
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  for (int i = 0; i < net_inputs_.size(); i++)
  {
      input_signals_[i].first = input_signals_[i].second->access(time);
      net_inputs_[i] = torch::from_blob(
            input_signals_[i].first.data(), {input_signals_[i].first.size()},
            options);
  }

  // Run the network given the input.
  network_result_ = module_.forward(net_inputs_);

  // Return the time required to run the network in ms.
  res = run_timer_.tac() * 1000.;

  return res;
}

void PyTorchEntity::warmup()
{
    // Run the network three times to warmup the computations.
    signal_run_network_(0);
    signal_run_network_(1);
    signal_run_network_(2);
}

dynamicgraph::Vector& PyTorchEntity::signal_callbacks(
        const std::string& output_name, dynamicgraph::Vector& res, const int& time)
{
    // Evaluate the network if needed.
    signal_run_network_(time);

    at::Tensor output;

    // If there is only a single result registered, then assume the model
    // returned a tensor.
    if (output_signal_names_.size() == 1)
    {
        output = network_result_.toTensor();
    } else {
        auto it = std::find(
            output_signal_names_.begin(), output_signal_names_.end(), output_name);
        if (it == output_signal_names_.end()) {
            throw std::runtime_error("PyTorchEntity: Output signal not found?");
        }
        int index = std::distance(output_signal_names_.begin(), it);

        // Expecting to get multiple arguments back.
        if (!network_result_.isTuple()) {
            throw std::runtime_error("PyTorchEntity: Expecting multiple output but module didn't return a tuple.");
        }

        output = network_result_.toTuple()->elements()[index].toTensor();
    }

    // Resize res if neded.
    int size = output.size(0);
    res.resize(size);

    // Copy the data to the output vector.
    auto output_a = output.accessor<double, 1>();
    for (int i = 0; i < size; i++)
    {
        res[i] = output_a[i];
    }

    return res;
}


} // namespace dg_pytorch
