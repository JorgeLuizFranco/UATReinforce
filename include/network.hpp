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

class DQLAgent {
private:
    torch::Device device;
    std::shared_ptr<NeuralNetwork> qNetwork;
    std::shared_ptr<NeuralNetwork> targetNetwork;
    ReplayBuffer replayBuffer;
    torch::optim::Adam optimizer;

    double gamma;
    double epsilon;
    double epsilonMin;
    double epsilonDecay;
    size_t stateSize;
    size_t actionSize;

public:
    DQLAgent(size_t stateSize, size_t actionSize, double gamma = 0.99,
             double epsilon = 1.0, double epsilonMin = 0.01, double epsilonDecay = 0.995, long long replayMemorySize = 10000);
    int getAction(const std::vector<double>& state);
    void train();
    void storeExperience(const std::vector<double>& state, int action, double reward,
                         const std::vector<double>& nextState, bool done);
    void syncTargetNetwork();
};