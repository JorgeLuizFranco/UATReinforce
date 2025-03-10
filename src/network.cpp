#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int actionSize, int time_steps, int input_channels)
    : action_size(actionSize),
      state_size(stateSize){

      conv_layers = torch::nn::Sequential(
        // First conv layer - preserve time dimension by using padding
        torch::nn::Conv3d(torch::nn::Conv3dOptions(input_channels, 8, {3, 3, 3}).padding({1, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(8),

        // Second conv layer - preserve time dimension
        torch::nn::Conv3d(torch::nn::Conv3dOptions(8, 16, {3, 3, 3}).padding({1, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(16),

        // Third conv layer - preserve time dimension
        torch::nn::Conv3d(torch::nn::Conv3dOptions(16, 32, {3, 3, 3}).padding({1, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(32)
      );


      decoder = torch::nn::Sequential(
        torch::nn::Conv3d(torch::nn::Conv3dOptions(32, 16, {3, 3, 3}).padding({1, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(16),

        torch::nn::Conv3d(torch::nn::Conv3dOptions(16, 8, {3, 3, 3}).padding({1, 1, 1})),
        torch::nn::ReLU(),
        torch::nn::BatchNorm3d(8),

        // Change to output 2 channels - 1 for mean and 1 for std_dev
        torch::nn::Conv3d(torch::nn::Conv3dOptions(8, 2, {3, 3, 3}).padding({1, 1, 1}))
      );

    // Rest of initialization remains the same
    for (auto& module : modules(/*include_self=*/false)) {
      if (auto* conv = module->as<torch::nn::Conv3d>()) {
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

  // Extract mean and std_dev from channels dimension
  auto mean = output.slice(1, 0, 1);     // First channel: mean
  auto std_dev = output.slice(1, 1, 2);  // Second channel: std deviation

  // Apply activations
  std_dev = torch::nn::functional::softplus(std_dev);

  // Reshape to incorporate time dimension (assuming time is dimension 2)
  // Each action gets its own distribution parameters across time steps
  mean = mean.squeeze(1);       // Remove channel dimension, now [batch, time, H, W]
  std_dev = std_dev.squeeze(1); // Remove channel dimension, now [batch, time, H, W]

  // Reshape to combine spatial dimensions and keep time separate
  mean = mean.flatten(2);       // Now [batch, time, spatial_features]
  std_dev = std_dev.flatten(2); // Now [batch, time, spatial_features]

  // If you need to ensure exactly 5 time steps
  if (mean.size(1) != 5) {
    // Either interpolate or pad/trim to get exactly 5 time steps
    mean = torch::nn::functional::interpolate(
      mean.unsqueeze(1), // Add channel dim for interpolation
      torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({5, mean.size(2)}))
        .mode(torch::kLinear)
    ).squeeze(1);

    std_dev = torch::nn::functional::interpolate(
      std_dev.unsqueeze(1), // Add channel dim for interpolation
      torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({5, std_dev.size(2)}))
        .mode(torch::kLinear)
    ).squeeze(1);
  }

  // Flatten completely for output
  mean = mean.flatten();
  std_dev = std_dev.flatten();

  return {mean, std_dev};
}