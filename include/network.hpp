#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <limits>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "airspace3d.hpp"

#include <torch/torch.h>

class NeuralNetwork : public torch::nn::Module {
private:
    torch::nn::Linear layer1, layer2, outputLayer;

public:
    NeuralNetwork(int stateSize, int hiddenSize, int actionSize);
    torch::Tensor forward(torch::Tensor x);
};

class ReplayBuffer {
private:
    std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> buffer;
    size_t maxSize;

public:
    ReplayBuffer(size_t size);
    void add(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
             torch::Tensor nextState, torch::Tensor done);
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
    sample(size_t batchSize);
    size_t size() const;
};

class Environment {
public:
    std::vector<double> state;
    int stepCount;
    int maxSteps;

    Environment(size_t stateSize, int maxSteps = 200);
    void reset();
    std::tuple<std::vector<double>, double, bool> step(int action);
    std::vector<double> getState() const;
};