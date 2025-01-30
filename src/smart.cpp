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

Smart::Smart(const Airspace3d& airspace, int seed, size_t stateSize, size_t actionSize, float learning_rate)
  : current_mission(airspace.random_mission(seed)),
    rng(seed),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    qNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
    optimizer(std::make_unique<torch::optim::Adam>(qNetwork->parameters(), torch::optim::AdamOptions(1e-3))),
    learning_rate(learning_rate),
    stateSize(stateSize),
    actionSize(actionSize)
{
  std::mt19937 rng(seed);

  current_mission = airspace.random_mission(rng());
  std::cout << "tenho que ir de " << current_mission.from.pos[0] << " " << current_mission.from.pos[1] << std::endl;
  std::cout << "para " << current_mission.to.pos[0] << " " << current_mission.to.pos[1] << std::endl;


  // deep q learning
  qNetwork->to(device);

  // airspace dimensions
  x = 10;
  y = 10;

  curr_state = std::vector<float>(x * y, 0.0);
  old_state = std::vector<float>(x*y, 0.0);
  last_action = std::vector<float>();
  rewards = std::vector<float>();

  target_time = 100;
  curr_time = 0;
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  using namespace uat::permit_public_status;
  curr_time++;

  last_action = getAction(curr_state);

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      Slot3d new_slot{{static_cast<uint_t>(i), static_cast<uint_t>(j), 0}};

      // Bidding
      std::visit(cool::compose{
        [](unavailable) { assert(false); },
        [](owned) { assert(false); },
        [&, slot = new_slot, t = curr_time](available) {
          bid(std::move(slot), t, last_action[i*x+j]);
        },
      }, status(new_slot, curr_time));
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
  // std::cout << "comprei " << location.pos[0] << " " << location.pos[1] << std::endl;
  // std::cout << "gastei " << v << " no tempo " << time << std::endl;

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

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << curr_state[i*10+j] << " ";
    }
    std::cout << std::endl;
  }
  
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

  if (target_time < curr_time) {
    target_time += 20;
  }

  fmt::print(stderr, msg + " {}\n", t);
  return stop;
}

std::vector<float> Smart::getAction(const std::vector<float>& state) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 1. Ensure gradient tracking
    qNetwork->train();  // Switch to train mode
    torch::AutoGradMode enable_grad(true);  // Enable gradient tracking

    // 2. Convert state to tensor with requires_grad
    torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat32))
                                .to(device)
                                .requires_grad_(true);

    // 3. Forward pass (now tracks gradients)
    auto [mean, log_std] = qNetwork->forward(stateTensor);

    auto std = torch::exp(0.5 * log_std);

    // // Sample and squash through sigmoid
    // auto noise = torch::randn(mean.sizes());
    // auto pre_squash = mean + std * noise;
    // auto bids = torch::sigmoid(pre_squash);  // Now in [0,1]

    // // Compute log probability with Jacobian correction
    // auto log_prob_pre = (-0.5 * (noise.pow(2) + std::log(2 * M_PI) + 2 * log_var)).sum();
    // auto log_jacobian = torch::log(bids * (1 - bids) + 1e-8).sum();  // Avoid log(0)
    // log_probs = log_prob_pre - log_jacobian;  // Account for sigmoid transform

    // Manual sampling from normal distribution
    torch::Tensor noise = torch::empty_like(mean);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    auto* noise_data = noise.data_ptr<float>();
    for (int i = 0; i < noise.numel(); i++) {
        noise_data[i] = dist(gen);
    }

    // Reparameterized sample
    torch::Tensor raw_action = mean + std * noise;

    // Sigmoid transformation
    torch::Tensor action = torch::sigmoid(raw_action);

    // Manual log probability calculation
    torch::Tensor log_prob = -0.5 * (
        torch::pow((raw_action - mean) / std, 2) +
        2 * log_std +
        torch::log(2 * M_PI * torch::ones_like(std))
    );

    // Jacobian correction for sigmoid
    log_prob -= torch::log(action * (1 - action) + 1e-8);
    log_prob = log_prob.sum();  // Sum across action dimensions

    // Convert to vector
    auto flat_bids = action.flatten().contiguous();
    return std::vector<float>(flat_bids.data_ptr<float>(),
                             flat_bids.data_ptr<float>() + flat_bids.numel());
}
