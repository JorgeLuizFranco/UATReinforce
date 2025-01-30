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
    torch::nn::Linear layer1, layer2, outputLayer, output_log_std;

public:
    NeuralNetwork(int stateSize, int hiddenSize, int actionSize);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
