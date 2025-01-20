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

// TODO
// - Guardar recompensas

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
  std::uniform_real_distribution<value_t> f{50.0, 150.0};
  std::uniform_real_distribution<> bid_value(0.0, 25.0);

  fundamental_ = f(rng);
  current_mission = airspace.random_mission(rng());

  // deep q learning
  qNetwork->to(device);
  targetNetwork->to(device);
  syncTargetNetwork();

  // airspace dimensions
  x = 10;
  y = 10;

  curr_state = std::vector<double>(x * y, 0.0);
  old_state = std::vector<double>(x*y, 0.0);
  last_action = std::vector<double>();
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  using namespace uat::permit_public_status;

  last_action = getAction(curr_state);

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      Slot3d new_slot{{static_cast<uint_t>(y), static_cast<uint_t>(x), 0}};

      // Bidding
      std::visit(cool::compose{
        [](unavailable) { assert(false); },
        [](owned) { assert(false); },
        [&, slot = new_slot, t = time](available) {
          bid(std::move(slot), t, last_action[i*x+y]);
        },
      }, status(new_slot, time));
    }
  }

  old_state.assign(curr_state.begin(), curr_state.end());
}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{
}

auto Smart::on_bought(const Slot3d& location, uat::uint_t time, uat::value_t v) -> void
{
  spent += v;

  keep_.insert({location, time});

  // Updating current state
  curr_state[location.pos[1] * x + location.pos[0]] = 1;

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
  std::queue<Slot3d> toVisit;
  std::unordered_set<Slot3d> visited;
  toVisit.push(current_mission.from);
  visited.insert(current_mission.from);
  uint_t min_dist = 1e9;

  while(!toVisit.empty()) {
      auto current = toVisit.front();
      toVisit.pop();
      
      min_dist = std::min(min_dist, current.distance(current_mission.to));
      if(min_dist == 0) {
          // storeExperience(old_state, last_action, 100, curr_state, true);
          fmt::print(stderr, "Smart agent has stopped at time {}\n", t);
          return true;
      }
      for(const auto& neighbor : current.neighbors()) {
          // Check if neighbor is in keep_
          // (time can be handled as needed)
          if(keep_.contains({neighbor, t}) &&
            !visited.count(neighbor)) {
              visited.insert(neighbor);
              toVisit.push(neighbor);
          }
      }
  }

  // storeExperience(old_state, last_action, -1, curr_state, false);
  fmt::print(stderr, "Smart agent did not finish mission yet at time {}\n", t);
  return false;
}

void Smart::syncTargetNetwork() {
    // Correct way to copy weights in LibTorch C++
    for (size_t i = 0; i < qNetwork->parameters().size(); ++i) {
        targetNetwork->parameters()[i].data().copy_(qNetwork->parameters()[i].data());
    }
}

std::vector<double> Smart::getAction(const std::vector<double>& state) {
    torch::Tensor stateTensor = torch::tensor(state,  torch::dtype(torch::kFloat64)).to(device);

    qNetwork->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor qValues = qNetwork->forward(stateTensor);

    std::vector<double> qValuesVec(qValues.data_ptr<double>(), qValues.data_ptr<double>() + qValues.numel());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < x*y; i++) {
      if (dis(gen) < epsilon) {
        qValuesVec[i] = dis(gen);
      }
    }

    return qValuesVec;

    // return qValues.argmax(0).item<int>();
}

void Smart::train() {
    if (replayBuffer.size() < 64) return;

    // Sample a minibatch from the replay buffer
    auto batch = replayBuffer.sample(64);

    // Separate experiences into tensors
    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (const auto& experience : batch) {
        states.push_back(std::get<0>(experience));
        actions.push_back(std::get<1>(experience));
        rewards.push_back(std::get<2>(experience));
        next_states.push_back(std::get<3>(experience));
        dones.push_back(std::get<4>(experience));
    }

    // Stack tensors to create batches and move to device
    torch::Tensor statesBatch = torch::stack(states).to(device);
    torch::Tensor actionsBatch = torch::stack(actions).to(device);
    torch::Tensor rewardsBatch = torch::stack(rewards).to(device);
    torch::Tensor nextStatesBatch = torch::stack(next_states).to(device);
    torch::Tensor donesBatch = torch::stack(dones).to(device);

    // Set network to training mode and reset gradients
    qNetwork->train();
    optimizer->zero_grad();

    // Compute current Q-values for the taken actions
    torch::Tensor currentQValues = qNetwork->forward(statesBatch).gather(1, actionsBatch.unsqueeze(1));

    // Compute next Q-values using the target network
    auto nextQValuesTuple = targetNetwork->forward(nextStatesBatch).max(1);
    torch::Tensor nextQValues = std::get<0>(nextQValuesTuple);

    // Compute target Q-values
    torch::NoGradGuard noGrad;
    // torch::Tensor nextQValues = targetNetwork->forward(nextStatesBatch).max(1).values;
    torch::Tensor targetQValues = rewardsBatch + (1 - donesBatch) * gamma * nextQValues;

    // Compute loss between current Q-values and target Q-values
    torch::Tensor loss = torch::mse_loss(currentQValues.squeeze(1), targetQValues);

    // Backpropagation
    loss.backward();
    optimizer->step();

    // Update epsilon for the epsilon-greedy policy
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
