#include "network.hpp"

NeuralNetwork::NeuralNetwork(int stateSize, int actionSize, int time_steps, int input_channels)
  : action_size(actionSize),
    state_size(stateSize)
{
  conv_layers = torch::nn::Sequential(
    // Increased filters
    torch::nn::Conv3d(torch::nn::Conv3dOptions(input_channels, 16, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(16),

    // Increased filters
    torch::nn::Conv3d(torch::nn::Conv3dOptions(16, 32, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(32),

    // Increased filters
    torch::nn::Conv3d(torch::nn::Conv3dOptions(32, 64, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(64),

    // Added layer
    torch::nn::Conv3d(torch::nn::Conv3dOptions(64, 128, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(128)
  );

  adaptive_pool = torch::nn::AdaptiveAvgPool3d(torch::nn::AdaptiveAvgPool3dOptions({5, 1, 1}));

  decoder = torch::nn::Sequential(
    // Adjusted input channels, increased filters
    torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(128, 64, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(64),

    // Increased filters
    torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(64, 32, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(32),

    // Added layer
    torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(32, 16, {5, 3, 3}).padding({2, 1, 1})),
    torch::nn::ReLU(),
    torch::nn::BatchNorm3d(16),

    // Adjusted input channels, change to output 2 channels
    torch::nn::ConvTranspose3d(torch::nn::ConvTranspose3dOptions(16, 2, {5, 3, 3}).padding({2, 1, 1}))
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
  std_dev = torch::nn::functional::softplus(std_dev) + 1e-6;
  mean = torch::nn::functional::relu(mean);

  // Reshape to incorporate time dimension
  // Each action gets its own distribution parameters across time steps
  mean = mean.squeeze(1);
  std_dev = std_dev.squeeze(1);

  // Reshape to combine spatial dimensions and keep time separate
  mean = mean.flatten(2);       // Now [batch, time, spatial_features]
  std_dev = std_dev.flatten(2); // Now [batch, time, spatial_features]

  // Ensure exactly 5 time steps
  if (mean.size(1) != 5) {
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

void NeuralNetwork::save_model(const std::string& filename) {
  torch::save(this->conv_layers, "conv_" + filename);
  torch::save(this->decoder, "decoder_" + filename);
  std::cout << "Model saved to " << filename << std::endl;
}

void NeuralNetwork::load_model(const std::string& filename) {
  try {
    torch::load(this->conv_layers, "conv_" + filename);
    torch::load(this->decoder, "decoder_" + filename);
    std::cout << "Model loaded from " << filename << std::endl;
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
  }
}