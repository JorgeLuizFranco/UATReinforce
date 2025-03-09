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
    torch::nn::Sequential conv_layers{nullptr};
    torch::nn::AdaptiveAvgPool3d adaptive_pool{nullptr};
    torch::nn::Sequential decoder{nullptr};
    int action_size;
    int state_size;

public:
    NeuralNetwork(int stateSize, int actionSize, int time_steps, int input_channels);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
