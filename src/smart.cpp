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

  // airspace dimensions
  x = 10;
  y = 10;

  curr_state = std::vector<double>(x * y, 0.0);
  old_state = std::vector<double>(x*y, 0.0);
  last_action = std::vector<double>();
  rewards = std::vector<double>();
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

std::vector<double> Smart::getAction(const std::vector<double>& state) {
    torch::Tensor stateTensor = torch::tensor(state,  torch::dtype(torch::kFloat64)).to(device);

    qNetwork->eval();
    torch::NoGradGuard noGrad;

    // Forward to get the mean and log var
    torch::Tensor raw_output = qNetwork->forward(stateTensor);
    auto split = torch::chunk(raw_output, 2, 0);

    // Create normal distribution
    auto mean = split[0].view({x, y});
    auto log_var = split[1].view({x, y});
    auto std = torch::exp(0.5 * log_var);

    // Sample and calculate log probabilities
    
    // Manual normal sampling with reparameterization trick
    auto noise = torch::randn(mean.sizes());
    auto bids = mean + std * noise;
    auto flat_bids = bids.flatten();
    flat_bids = flat_bids.contiguous().to(torch::kDouble);
    
    // Manually calculate log probability
    log_probs = (-0.5 * (noise.pow(2) + std::log(2 * M_PI) + 2 * log_var)).sum();
        
    return std::vector<double>(flat_bids.data_ptr<double>(), flat_bids.data_ptr<double>() + flat_bids.numel());
}

