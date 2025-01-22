#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int hiddenSize, int actionSize)
    : layer1(register_module("layer1", torch::nn::Linear(stateSize, hiddenSize))),
      layer2(register_module("layer2", torch::nn::Linear(hiddenSize, hiddenSize))),
      outputLayer(register_module("outputLayer", torch::nn::Linear(hiddenSize, 2*actionSize))) {
}

torch::Tensor NeuralNetwork::forward(torch::Tensor x) {
    x = torch::relu(layer1(x));
    x = torch::relu(layer2(x));
    x = outputLayer(x);
    return x;
}

// Experience Replay Buffer
ReplayBuffer::ReplayBuffer(size_t size) : maxSize(size) {}

void ReplayBuffer::add(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
                       torch::Tensor nextState, torch::Tensor done) {
    if (buffer.size() >= maxSize) {
        buffer.pop_front();
    }
    buffer.push_back(std::make_tuple(state, action, reward, nextState, done));
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
ReplayBuffer::sample(size_t batchSize) {
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch;
    std::random_device rd;
    std::mt19937 gen(rd());
    if (buffer.size() == 0) return batch;  // Handle empty buffer case
    std::uniform_int_distribution<> dis(0, buffer.size() - 1);

    for (size_t i = 0; i < std::min(batchSize, buffer.size()); ++i) {
        batch.push_back(buffer[dis(gen)]);
    }
    return batch;
}

size_t ReplayBuffer::size() const {
    return buffer.size();
}
