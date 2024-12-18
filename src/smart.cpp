#include <cool/compose.hpp>
#include <cool/indices.hpp>
#include <jules/base/numeric.hpp>
#include <range/v3/view/transform.hpp>

#include <cstdio>
#include <iterator>
#include <cassert>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include <variant>

#include "smart.hpp"

#include <fmt/core.h>

using namespace uat;

Smart::Smart(const Airspace3d& airspace, int seed, size_t stateSize, size_t actionSize, double gamma,
                   double epsilon, double epsilonMin, double epsilonDecay, long long replayMemorySize)
  : current_mission(airspace.random_mission(seed)),
    rng(seed),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    qNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
    targetNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
    replayBuffer(replayMemorySize),
    optimizer(std::make_unique<torch::optim::Adam>(qNetwork->parameters(), torch::optim::AdamOptions(1e-3))),
    gamma(gamma),
    epsilon(epsilon),
    epsilonMin(epsilonMin),
    epsilonDecay(epsilonDecay),
    stateSize(stateSize),
    actionSize(actionSize)
{
  std::mt19937 rng(seed);

  std::uniform_real_distribution<> dist;
  current_mission = airspace.random_mission(rng());

  // deep q learning
  qNetwork->to(device);
  targetNetwork->to(device);
  syncTargetNetwork();
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{

}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{

}

auto Smart::on_bought(const Slot3d& location, uat::uint_t time, uat::value_t v) -> void
{
  spent += v;

  keep_.insert({location, time});
  // check whether mission has been completed
  // then starts a new mission
  // mission_ = space.random_mission(rng());
}

auto Smart::on_sold(const Slot3d&, uat::uint_t, uat::value_t v) -> void
{
  spent -= v;
}

auto Smart::stop(uat::uint_t t, int) -> bool
{
  fmt::print(stderr, "Smart agent has stopped at time {}\n", t);
  return true;
}

void Smart::syncTargetNetwork() {
    // Correct way to copy weights in LibTorch C++
    for (size_t i = 0; i < qNetwork->parameters().size(); ++i) {
        targetNetwork->parameters()[i].data().copy_(qNetwork->parameters()[i].data());
    }
}

int Smart::getAction(const std::vector<double>& state) {
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

void Smart::train() {
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
    optimizer->zero_grad();

    torch::Tensor currentQValues = qNetwork->forward(statesBatch).gather(1, actionsBatch.unsqueeze(1));

    auto nextQValuesTuple = targetNetwork->forward(nextStatesBatch).max(1);
    torch::Tensor nextQValues = std::get<0>(nextQValuesTuple);

    torch::NoGradGuard noGrad;
    // torch::Tensor nextQValues = targetNetwork->forward(nextStatesBatch).max(1).values;
    torch::Tensor targetQValues = rewardsBatch + (1 - donesBatch) * gamma * nextQValues;

    torch::Tensor loss = torch::mse_loss(currentQValues.squeeze(1), targetQValues);

    loss.backward();
    optimizer->step();

    epsilon = std::max(epsilonMin, epsilon * epsilonDecay);

    static int updateCounter = 0;
    if (++updateCounter % 100 == 0) {
        syncTargetNetwork();
    }
}

void Smart::storeExperience(const std::vector<double>& state, int action, double reward,
                             const std::vector<double>& nextState, bool done) {

    torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat64)).to(device);
    torch::Tensor actionTensor = torch::tensor({(long)action}).to(device);  // Actions as tensors
    torch::Tensor rewardTensor = torch::tensor({reward}).to(device);
    torch::Tensor nextStateTensor = torch::tensor(nextState, torch::dtype(torch::kFloat64)).to(device);
    torch::Tensor doneTensor = torch::tensor({done}).to(device);

    replayBuffer.add(stateTensor, actionTensor, rewardTensor, nextStateTensor, doneTensor);
}
