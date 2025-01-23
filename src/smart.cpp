#include <ATen/ops/chunk.h>
#include <c10/core/ScalarType.h>
#include <cool/compose.hpp>
#include <cool/indices.hpp>
#include <jules/base/numeric.hpp>
#include <range/v3/view/transform.hpp>

#include <cstdio>
#include <iterator>
#include <cassert>
#include <limits>
#include <random>
#include <torch/types.h>
#include <unordered_set>
#include <utility>
#include <variant>
#include <string>

#include "smart.hpp"

#include <fmt/core.h>

using namespace uat;

Smart::Smart(const Airspace3d& airspace, int seed, size_t stateSize, size_t actionSize, double gamma,
                   double epsilon, double epsilonMin, double epsilonDecay, long long replayMemorySize, float learning_rate)
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
    learning_rate(learning_rate),
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

  // airspace dimensions
  x = 10;
  y = 10;

  curr_state = std::vector<float>(x * y, 0.0);
  old_state = std::vector<float>(x*y, 0.0);
  last_action = std::vector<float>();
  rewards = std::vector<float>();
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  using namespace uat::permit_public_status;

  last_action = getAction(curr_state);
  for (const auto& loc : last_action) {
    std::cout << loc << " ";
  }
  std::cout << std::endl;

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

  double reward = -1;
  std::string msg = "Smart agent did not finish mission yet at time";
  bool stop = false;

  while(!toVisit.empty()) {
      auto current = toVisit.front();
      toVisit.pop();

      min_dist = std::min(min_dist, current.distance(current_mission.to));

      if(min_dist == 0) {
          reward = 100;
          msg = "Smart agent has stopped at time";
          stop =  true;
          break;
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

  // Updating network
  // Backpropagation with proper gradient retention
  if (log_probs.requires_grad()) {
    auto loss = -log_probs * reward;
    optimizer->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_value_(qNetwork->parameters(), 1.0);  // Clip gradients to L2 norm of 1.0
    optimizer->step();
  }

  // Updating the slots we have for next round of bids
  for (const auto [location, time] : keep_) {
    if (time <= t) {
      curr_state[location.pos[1] * x + location.pos[0]] = 0;
    }
  }

  fmt::print(stderr, msg + " {}\n", t);
  return stop;
}

std::vector<float> Smart::getAction(const std::vector<float>& state) {
    // 1. Ensure gradient tracking
    qNetwork->train();  // Switch to train mode
    torch::AutoGradMode enable_grad(true);  // Enable gradient tracking

    // 2. Convert state to tensor with requires_grad
    torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat32))
                                .to(device)
                                .requires_grad_(true);

    // 3. Forward pass (now tracks gradients)
    torch::Tensor raw_output = qNetwork->forward(stateTensor);
    // Split into mean and log_var
    auto split = torch::chunk(raw_output, 2, 0);

    // Constrain mean to [0,1] using sigmoid
    auto mean = torch::sigmoid(split[0].view({x, y}));

    // Constrain log_var to prevent explosion
    auto log_var = torch::clamp(split[1].view({x, y}), -10.0, 5.0);
    auto std = torch::exp(0.5 * log_var);

    // Sample and squash through sigmoid
    auto noise = torch::randn(mean.sizes());
    auto pre_squash = mean + std * noise;
    auto bids = torch::sigmoid(pre_squash);  // Now in [0,1]

    // Compute log probability with Jacobian correction
    auto log_prob_pre = (-0.5 * (noise.pow(2) + std::log(2 * M_PI) + 2 * log_var)).sum();
    auto log_jacobian = torch::log(bids * (1 - bids) + 1e-8).sum();  // Avoid log(0)
    log_probs = log_prob_pre - log_jacobian;  // Account for sigmoid transform

    // Convert to vector
    auto flat_bids = bids.flatten().contiguous();
    return std::vector<float>(flat_bids.data_ptr<float>(),
                             flat_bids.data_ptr<float>() + flat_bids.numel());
}
