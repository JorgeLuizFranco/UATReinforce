#include "network.hpp"

// Neural Network implementation using PyTorch
NeuralNetwork::NeuralNetwork(int stateSize, int hiddenSize, int actionSize)
    : layer1(register_module("layer1", torch::nn::Linear(stateSize, hiddenSize))),
      layer2(register_module("layer2", torch::nn::Linear(hiddenSize, hiddenSize))),
      outputLayer(register_module("outputLayer", torch::nn::Linear(hiddenSize, actionSize))) {
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

// Deep Q-Learning Agent
DQLAgent::DQLAgent(size_t stateSize, size_t actionSize, double gamma,
                   double epsilon, double epsilonMin, double epsilonDecay, long long replayMemorySize)
    :   device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        qNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
        targetNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
        replayBuffer(replayMemorySize),
        optimizer(qNetwork->parameters(), torch::optim::AdamOptions(1e-3)),
        gamma(gamma), epsilon(epsilon), epsilonMin(epsilonMin),
        epsilonDecay(epsilonDecay), stateSize(stateSize), actionSize(actionSize)
{
    qNetwork->to(device);
    targetNetwork->to(device);
    syncTargetNetwork();
}

void DQLAgent::syncTargetNetwork() {
    // Correct way to copy weights in LibTorch C++
    for (size_t i = 0; i < qNetwork->parameters().size(); ++i) {
        targetNetwork->parameters()[i].data().copy_(qNetwork->parameters()[i].data());
    }
}

int DQLAgent::getAction(const std::vector<double>& state) {
    torch::Tensor stateTensor = torch::tensor(state,  torch::dtype(torch::kFloat64)).to(device);

    qNetwork->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor qValues = qNetwork->forward(stateTensor);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < epsilon) {
        std::uniform_int_distribution<> actionDis(0, actionSize - 1);
        return actionDis(gen);
    }

    return qValues.argmax(0).item<int>();
}

void DQLAgent::train() {
    if (replayBuffer.size() < 64) return;

    auto batch = replayBuffer.sample(64);

    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (const auto& experience : batch) {
        states.push_back(std::get<0>(experience));
        actions.push_back(std::get<1>(experience));
        rewards.push_back(std::get<2>(experience));
        next_states.push_back(std::get<3>(experience));
        dones.push_back(std::get<4>(experience));
    }

    torch::Tensor statesBatch = torch::stack(states).to(device);
    torch::Tensor actionsBatch = torch::stack(actions).to(device);
    torch::Tensor rewardsBatch = torch::stack(rewards).to(device);
    torch::Tensor nextStatesBatch = torch::stack(next_states).to(device);
    torch::Tensor donesBatch = torch::stack(dones).to(device);



    qNetwork->train();
    optimizer.zero_grad();

    torch::Tensor currentQValues = qNetwork->forward(statesBatch).gather(1, actionsBatch.unsqueeze(1)); 

    auto nextQValuesTuple = targetNetwork->forward(nextStatesBatch).max(1);
    torch::Tensor nextQValues = std::get<0>(nextQValuesTuple);

    torch::NoGradGuard noGrad;
    // torch::Tensor nextQValues = targetNetwork->forward(nextStatesBatch).max(1).values;
    torch::Tensor targetQValues = rewardsBatch + (1 - donesBatch) * gamma * nextQValues;

    torch::Tensor loss = torch::mse_loss(currentQValues.squeeze(1), targetQValues);

    loss.backward();
    optimizer.step();

    epsilon = std::max(epsilonMin, epsilon * epsilonDecay);

    static int updateCounter = 0;
    if (++updateCounter % 100 == 0) {
        syncTargetNetwork();
    }
}

void DQLAgent::storeExperience(const std::vector<double>& state, int action, double reward,
                             const std::vector<double>& nextState, bool done) {

    torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat64)).to(device);
    torch::Tensor actionTensor = torch::tensor({(long)action}).to(device);  // Actions as tensors
    torch::Tensor rewardTensor = torch::tensor({reward}).to(device);
    torch::Tensor nextStateTensor = torch::tensor(nextState, torch::dtype(torch::kFloat64)).to(device);
    torch::Tensor doneTensor = torch::tensor({done}).to(device);

    replayBuffer.add(stateTensor, actionTensor, rewardTensor, nextStateTensor, doneTensor);
}

Environment::Environment(size_t stateSize, int maxSteps) : state(stateSize), stepCount(0), maxSteps(maxSteps) {
    reset();
}

void Environment::reset() {
    stepCount = 0;
    std::fill(state.begin(), state.end(), 0.0);
}

std::tuple<std::vector<double>, double, bool> Environment::step(int action) {
    stepCount++;
    double reward = 0.0;
    if (action == 0) {
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] -= 0.1;
        }
    } else {
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] += 0.1;
        }
    }

    if (stepCount >= maxSteps) {
        if (state[0] > 1.0) {
            reward = 10.0;
        }
        return std::make_tuple(state, reward, true);
    }

    return std::make_tuple(state, reward, false);
}

std::vector<double> Environment::getState() const {
    return state;
}