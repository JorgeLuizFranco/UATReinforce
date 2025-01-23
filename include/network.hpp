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