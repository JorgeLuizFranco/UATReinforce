#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int hiddenSize, int actionSize)
    : layer1(register_module("layer1", torch::nn::Linear(stateSize, hiddenSize))),
      layer2(register_module("layer2", torch::nn::Linear(hiddenSize, hiddenSize))),
      outputLayer(register_module("outputLayer", torch::nn::Linear(hiddenSize, 2*actionSize))) {

    torch::nn::init::xavier_normal_(layer1->weight);
    torch::nn::init::constant_(layer1->bias, 0.0);
    torch::nn::init::xavier_uniform_(layer2->weight, 1.0);  // Gain=1.0 for sigmoid
    torch::nn::init::constant_(layer2->bias, 0.0);
    }

torch::Tensor NeuralNetwork::forward(torch::Tensor x) {
    x = torch::relu(layer1(x));
    x = torch::relu(layer2(x));
    x = outputLayer(x);
    return x;
}
