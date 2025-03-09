#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int actionSize, int time_steps, int input_channels)
    : action_size(actionSize),
      state_size(stateSize){

      conv_layers = torch::nn::Sequential(
        // First conv layer
        torch::nn::Conv3d(torch::nn::Conv3dOptions(input_channels, 8, {2, 3, 3}).padding({0, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(8),

        // Second conv layer
        torch::nn::Conv3d(torch::nn::Conv3dOptions(8, 16, {2, 3, 3}).padding({0, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(16),

        // Third conv layer
        torch::nn::Conv3d(torch::nn::Conv3dOptions(16, 32, {2, 3, 3}).padding({0, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(32)
      );

      adaptive_pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions({1, 1, 1}));

      decoder = torch::nn::Sequential(
        torch::nn::Conv3d(torch::nn::Conv3dOptions(32, 16, {1, 3, 3}).padding({0, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(16),

        torch::nn::Conv3d(torch::nn::Conv3dOptions(16, 8, {1, 3, 3}).padding({0, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(8),

        // Change to output 10 channels - 5 for mean and 5 for std_dev
        torch::nn::Conv3d(torch::nn::Conv3dOptions(8, 10, {1, 3, 3}).padding({0, 1, 1}))
      );

    for (auto& module : modules(/*include_self=*/false)) {
      if (auto* conv = module->as<torch::nn::Conv3d>()) {
          // Kaiming initialization for convolutional layers (He initialization)
          torch::nn::init::kaiming_normal_(
              conv->weight, /*a=*/0, /*mode=*/torch::kFanOut, /*nonlinearity=*/torch::kReLU);
          if (conv->bias.defined()) {
              torch::nn::init::constant_(conv->bias, 0.0);
          }
      } else if (auto* bn = module->as<torch::nn::BatchNorm3d>()) {
          if (bn->weight.defined()) {
              torch::nn::init::constant_(bn->weight, 1.0);
          }
          if (bn->bias.defined()) {
              torch::nn::init::constant_(bn->bias, 0.0);
          }
      }
    }

}

std::tuple<torch::Tensor, torch::Tensor> NeuralNetwork::forward(torch::Tensor x) {
  auto features = conv_layers->forward(x);
  auto output = decoder->forward(features);

  if (output.size(2) > 1) {
    output = output.slice(2, output.size(2)-1, output.size(2));
  }

  // Split 10 channels into two groups of 5 for mean and std
  auto mean = output.slice(1, 0, 5);        // First 5 channels: mean
  auto std_dev = output.slice(1, 5, 10);    // Second 5 channels: std deviation

  // Apply activations
  std_dev = torch::nn::functional::softplus(std_dev);

  // std::cout << "Mean shape: " << mean.sizes() << " Std shape: " << std_dev.sizes() << std::endl;

  mean = mean.flatten();
  std_dev = std_dev.flatten();

  // std::cout << "Flattened mean shape: " << mean.sizes() << " std shape: " << std_dev.sizes() << std::endl;


  return {mean, std_dev};
}