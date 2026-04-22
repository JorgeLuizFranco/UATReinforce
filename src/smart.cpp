#include <ATen/ops/chunk.h>
#include <c10/core/ScalarType.h>
#include <cool/compose.hpp>
#include <cool/indices.hpp>
#include <cool/ccreate.hpp>
#include <jules/base/numeric.hpp>

#include <cstdio>
#include <iterator>
#include <cassert>
#include <limits>
#include <random>
#include <torch/serialize/input-archive.h>
#include <torch/types.h>
#include <unordered_set>
#include <utility>
#include <variant>
#include <string>
#include <fstream>

#include "smart.hpp"

#include <fmt/core.h>

using namespace uat;

Smart::Smart(const Airspace2d& airspace, int seed, size_t stateSize, size_t actionSize, float learning_rate)
  : current_mission(airspace.random_mission(seed)),
    rng(seed),
    space(airspace),
    device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
    qNetwork(std::make_shared<NeuralNetwork>(stateSize, actionSize, 5, 1)),
    optimizer(std::make_unique<torch::optim::Adam>(qNetwork->parameters(), torch::optim::AdamOptions(1e-3))),
    learning_rate(learning_rate),
    stateSize(stateSize),
    actionSize(actionSize)
{
  current_mission = airspace.random_mission(rng());

  //loading model if exists
  qNetwork->to(device);
  std::ifstream f("q_network.pt");
  if (f.good()) {
    qNetwork->load_model("q_network.pt");
  }

  // airspace dimensions
  x = 15;
  y = 15;

  // reward function parameters
  faturamento_bruto = 110.0;
  rewards = std::vector<float>();
  log_probs = std::vector<torch::Tensor>();
  gamma = 0.99;

  // agent states
  curr_state = std::vector<float>(x * y, 0.0);
  old_state = std::vector<float>(x*y, 0.0);

  curr_time = 0;
}

auto Smart::bid_phase(uat::uint_t, uat::bid_fn bid, uat::permit_public_status_fn status, int) -> void
{
  using namespace uat::permit_public_status;
  curr_time++;

  // Do not bid if can't achieve mission in all possible times
  if (!can_finish_mission(curr_time, status)) {
    return;
  }

  auto state_t1 = calculate_dist(curr_time, status);
  auto state_t2 = calculate_dist(curr_time+1, status);
  auto state_t3 = calculate_dist(curr_time+2, status);
  auto state_t4 = calculate_dist(curr_time+3, status);
  auto state_t5 = calculate_dist(curr_time+4, status);

  std::vector<float> state = stack_states(
    {&state_t1, &state_t2, &state_t3, &state_t4, &state_t5}
  );

  auto bid_values = getAction(state);
  for (int plus_time = 0; plus_time < 5; plus_time++) {
    for (int i = 0; i < x; i++) {
      for (int j = 0; j < y; j++) {
        Slot2d new_slot{{static_cast<uint_t>(i), static_cast<uint_t>(j)}, {static_cast<uint_t>(x), static_cast<uint_t>(y)}};

        // Bidding
        std::visit(cool::compose{
          [](unavailable) { /* do nothing */ },
          [](owned) { /* do nothing */ },
          [&, slot = new_slot, t = curr_time+plus_time](available) {
            bid(std::move(slot), t, bid_values[plus_time*x*y + i*x + j]);
          },
        }, status(new_slot, curr_time+plus_time));
      }
    }
  }

  old_state.assign(curr_state.begin(), curr_state.end());
}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{
}

auto Smart::on_bought(const Slot2d& location, uat::uint_t time, uat::value_t v) -> void
{
  spent += v;
  keep_.insert({location, time});

  // Updating current state
  curr_state[location.pos[0] * x + location.pos[1]] = 1;
}

auto Smart::on_sold(const Slot2d& location, uat::uint_t, uat::value_t v) -> void
{
  spent -= v;
}

auto Smart::stop(uat::uint_t t, int) -> bool
{
  auto [msg, reward, stop] = can_achieve_mission(t);
  rewards.push_back(reward);

  if (stop) {
    back_propagation();
    episode_counter++;
    log_probs.clear();
    rewards.clear();
    std::fill(curr_state.begin(), curr_state.end(), 0);
    std::fill(old_state.begin(), old_state.end(), 0);
    keep_.clear();
    spent = 0;
    current_mission = space.random_mission(rng());
    return true;
  }

  clean_states(t);
  return false;
}

std::vector<float> Smart::getAction(const std::vector<float>& state) {
  qNetwork->eval();
  torch::AutoGradMode enable_grad(true);

  // Convert state to tensor
  int time_stamps = 5;
  torch::Tensor stateTensor = torch::tensor(state).reshape({1, 1, time_stamps, x, y});

  // Forward pass
  auto [log_mean, log_std] = qNetwork->forward(stateTensor);

  // Calculating std and noise to sample later
  auto std = torch::exp(log_std);
  auto mean = torch::exp(log_mean);
  torch::Tensor noise = torch::randn_like(mean);

  // Sampling action
  torch::Tensor action = mean + std * noise;

  // Log probability calculation
  torch::Tensor log_prob = -0.5 * (
    torch::sum(torch::pow(noise, 2), -1) +
    2 * torch::sum(log_std, -1) +
    mean.size(-1) * std::log(2 * M_PI)
  );

  // Store log_prob
  log_probs.push_back(log_prob);

  // Converting tensor action to vector
  auto detached_action = action.detach();
  auto flat_bids = detached_action.flatten().contiguous();

  return std::vector<float>(flat_bids.data_ptr<float>(),
                            flat_bids.data_ptr<float>() + flat_bids.numel());
}

torch::Tensor Smart::compute_returns() {
  if (rewards.empty()) {
    return torch::zeros({0}, torch::kFloat32);
  }

  std::vector<float> discounted;
  float R = 0;

  // Iterate through rewards in reverse order
  for (auto it = rewards.rbegin(); it != rewards.rend(); ++it) {
      R = *it + gamma * R;
      discounted.insert(discounted.begin(), R);
  }

  // Convert vector to torch tensor
  return torch::tensor(discounted, torch::kFloat32);
}

std::tuple<std::string, float, bool> Smart::can_achieve_mission(uint_t t) {
  std::queue<Slot2d> toVisit;
  std::unordered_set<Slot2d> visited;

  toVisit.push(current_mission.from);
  visited.insert(current_mission.from);

  uint_t min_dist = 1e9;
  float reward = -spent;
  std::string msg = "Smart agent did not finish mission yet at time";
  bool stop = false;
  int num_slots_used = 1;

  // BFS algorithm
  while(!toVisit.empty()) {
    auto current = toVisit.front();
    toVisit.pop();

    min_dist = std::min(min_dist, current.distance(current_mission.to));

    if (min_dist == 0) {
      reward += faturamento_bruto * current_mission.from.distance(current_mission.to);
      msg = "Smart agent has stopped at time";
      stop = true;
      return {msg, reward, stop};
    }

    for(const auto& neighbor : current.neighbors()) {
      if(keep_.contains({neighbor, t}) &&
        !visited.count(neighbor)) {
          visited.insert(neighbor);
          toVisit.push(neighbor);
          num_slots_used++;
      }
    }
  }

  return {msg, reward, stop};
}

void Smart::clean_states(uint_t t) {
  // Updating the slots we have for next round of bids
  for (const auto [location, time] : keep_) {
    if (time <= t) {
      curr_state[location.pos[1] * x + location.pos[0]] = 0;
    }
  }
}

void Smart::back_propagation() {
  qNetwork->train();
  torch::AutoGradMode enable_grad(true);

  // Computing future returns
  torch::Tensor returns_tensor = compute_returns();

  // Normalizing returns
  returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8);

  // Calculate loss
  torch::Tensor loss = torch::zeros({}, torch::kFloat32);
  for (size_t t = 0; t < log_probs.size(); ++t) {
      loss += -log_probs[t] * returns_tensor[t];
  }

  // Backpropagation
  optimizer->zero_grad();
  loss.backward();
  optimizer->step();
}

std::vector<float> Smart::calculate_dist(uint_t time, uat::permit_public_status_fn status) {
  using namespace uat::permit_public_status;
  auto from = current_mission.from;
  auto to = current_mission.to;
  std::vector<float> full_dist(x*y, x+y);

  std::queue<Slot2d> q;
  auto short_path = from.shortest_path(to, rng());
  for (const auto&p : short_path) {
    auto idx = p.pos[0]*x + p.pos[1];
    full_dist[idx] = 1.0f;
    q.push(p);
  }

  while(!q.empty()) {
    auto current = q.front(); q.pop();
    float curr_dist = full_dist[current.pos[0]*x + current.pos[1]];

    for(const auto& neighbor : current.neighbors()) {
      int idx = neighbor.pos[0]*x + neighbor.pos[1];

      if (full_dist[idx] > curr_dist+1) {
        full_dist[idx] = curr_dist+1;
        q.push(neighbor);
      }
    }
  }

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      Slot2d new_slot{{static_cast<uint_t>(i), static_cast<uint_t>(j)}, {static_cast<uint_t>(x), static_cast<uint_t>(y)}};
      if (std::holds_alternative<unavailable>(status(new_slot, time))) {
        full_dist[i*x+j] = 1.0/(x+y);
      }
      else if (std::holds_alternative<owned>(status(new_slot, time))) {
        full_dist[i*x+j] = 0.0;
      }
      else {
        full_dist[i*x+j] = 1.0/(full_dist[i*x+j]);
      }
    }
  }

  return full_dist;
}

bool Smart::can_finish_mission(uint_t curr_time, uat::permit_public_status_fn status) {
  using namespace uat::permit_public_status;
  int count = 0;
  for (int i = 0; i < time_steps; i++) {
    auto slot_status = status(current_mission.to, curr_time+i);
    if (std::holds_alternative<unavailable>(slot_status)) {
      count++;
    }
  }

  return count == time_steps;
}

std::vector<float> Smart::stack_states(const std::vector<const std::vector<float>*>& states) {
  std::vector<float> combined;
  combined.reserve(x * y * states.size());
  for (const auto* state : states) {
    for (int i = 0; i < x; i++) {
      for (int j = 0; j < y; j++) {
        combined.push_back((*state)[i * x + j]);
      }
    }
  }
  return combined;
}