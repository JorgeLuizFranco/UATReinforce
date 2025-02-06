#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int hiddenSize, int actionSize)
    : layer1(register_module("layer1", torch::nn::Linear(stateSize, hiddenSize))),
      layer2(register_module("layer2", torch::nn::Linear(hiddenSize, hiddenSize))),
      outputLayer(register_module("outputLayer", torch::nn::Linear(hiddenSize, actionSize))),
      output_log_std(register_module("output_log_std", torch::nn::Linear(hiddenSize, actionSize))) {

    torch::nn::init::kaiming_normal_(layer1->weight);
    torch::nn::init::constant_(layer1->bias, 0.0);
    torch::nn::init::kaiming_normal_(layer2->weight);
    torch::nn::init::constant_(layer2->bias, 0.0);


    // Initialize the outputLayer with moderate values.
    torch::nn::init::kaiming_normal_(outputLayer->weight);
    torch::nn::init::constant_(outputLayer->bias, 0.0);

    // For output_log_std, it is common to initialize the bias to a small negative value
    // so that the standard deviation starts around exp(-0.5) ~ 0.6
    torch::nn::init::constant_(output_log_std->weight, 0.0);
    torch::nn::init::constant_(output_log_std->bias, -0.5);
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::forward(torch::Tensor x) {
    x = torch::relu(layer1(x));
    x = torch::relu(layer2(x));
    auto mean = torch::softplus(outputLayer(x));
    auto log_std = output_log_std(x);
    return {mean, log_std};
}
