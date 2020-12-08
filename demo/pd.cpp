#include <torch/script.h> // One-stop header.

#include <Eigen/Dense>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  int size = 6;
  torch::Tensor kp = 5. * torch::ones(1.);

  Eigen::VectorXd des_position;
  des_position.resize(size);
  des_position << 1., 2., 3., 4., 5., 6.;

  Eigen::Ref<Eigen::VectorXd> ref_des_position = des_position;

  Eigen::VectorXd position;
  position.resize(size);
  position << 0., 1., 2., 3., 4., 4.;

  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor des_position_tensor = torch::from_blob(
      ref_des_position.data(), {ref_des_position.size()}, options);
  torch::Tensor position_tensor = torch::from_blob(
      position.data(), {position.size()}, options);


  std::cout << "des_position: " << des_position << std::endl;
  std::cout << "position: " << position << std::endl;

  std::cout << "des_position_tensor: " << des_position_tensor << std::endl;
  std::cout << "position_tensor: " << position_tensor << std::endl;

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(kp);
  inputs.push_back(des_position_tensor);
  inputs.push_back(position_tensor);

  at::Tensor output;
  for (int j = 0; j < 2; j++) {
    // Execute the model and turn its output into a tensor.
    torch::jit::IValue res = module.forward(inputs);
    if (res.isTensorList()) {
      std::cout << "Unpacking tensorList" << std::endl;
      auto list = res.toTensorList();
      output = list[0];
    } else if (res.isTuple()) {
      std::cout << "Unpacking tuple" << std::endl;
      auto tuple = res.toTuple();
      output = tuple->elements()[0].toTensor();
    } else {
      std::cout << "Unpacking tensor" << std::endl;
      output = res.toTensor();
    }

    printf("output.data()=%p\n", output.data());
    printf("output.size()=%d\n", output.size(0));
  }
  auto output_a = output.accessor<double, 1>();

  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  for (int i = 0; i < size; i++) {
    std::cout << output_a[i] << " ";
  }
  std::cout << '\n';

  std::cout << "ok\n";
}
