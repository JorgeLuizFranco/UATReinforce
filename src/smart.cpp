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
#include <torch/serialize/input-archive.h>
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
    space(airspace),
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
  log_probs = std::vector<torch::Tensor>();
  full_dist = std::vector<float>(x*y, 1e9f);
  gamma = 0.99;

  curr_time = 0;
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  using namespace uat::permit_public_status;
  curr_time++;

  calculate_dist(curr_time, status);

  // Creating state vector to represent slots that I own + distances from shortest path
  std::vector<float> state;
  state.reserve(curr_state.size() + full_dist.size());
  state.insert(state.end(), curr_state.begin(), curr_state.end());
  state.insert(state.end(), full_dist.begin(), full_dist.end());

  last_action = getAction(state);

  // std::cout << "Iniciando ofertas do leilao no tempo " << curr_time << std::endl;
  // for (int i = 0; i < x; i++) {
  //   for (int j = 0; j < y; j++) {
  //     std::cout << last_action[i*x+j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

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
  curr_state[location.pos[0] * x + location.pos[1]] = 1;

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
  auto [msg, reward, stop] = can_achieve_mission(t);
  rewards.push_back(reward);

  if (stop) {
    back_propagation();

    log_probs.clear();
    rewards.clear();
    std::fill(curr_state.begin(), curr_state.end(), 0);
    std::fill(old_state.begin(), old_state.end(), 0);
    std::fill(full_dist.begin(), full_dist.end(), 1e9f);
    keep_.clear();
    spent = 0;

    current_mission = space.random_mission(rng());
    std::cout << "Tenho que ir de " << current_mission.from.pos[0] << " " << current_mission.from.pos[1] << std::endl;
    std::cout << "Para " << current_mission.to.pos[0] << " " << current_mission.to.pos[1] << std::endl;
  }

  clean_states(t);

  // fmt::print(stderr, msg + " {}\n", t);
  return false;
}

std::vector<float> Smart::getAction(const std::vector<float>& state) {
  qNetwork->eval();
  torch::AutoGradMode enable_grad(true);

  // Convert state to tensor
  torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat32))
                              .to(device);

  // Forward pass
  auto [mean, log_std] = qNetwork->forward(stateTensor);

  // Calculating std and noise to sample later
  auto std = torch::exp(log_std).requires_grad_(true);
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
  std::queue<Slot3d> toVisit;
  std::unordered_set<Slot3d> visited;

  toVisit.push(current_mission.from);
  visited.insert(current_mission.from);

  uint_t min_dist = 1e9;
  float reward = -1;
  std::string msg = "Smart agent did not finish mission yet at time";
  bool stop = false;

  // BFS algorithm
  while(!toVisit.empty()) {
    auto current = toVisit.front();
    toVisit.pop();

    min_dist = std::min(min_dist, current.distance(current_mission.to));

    if (min_dist == 0) {
      reward = 100.0f;
      msg = "Smart agent has stopped at time";
      stop = true;
      break;
    }

    for(const auto& neighbor : current.neighbors()) {
      if(keep_.contains({neighbor, t}) &&
        !visited.count(neighbor)) {
          visited.insert(neighbor);
          toVisit.push(neighbor);
      }
    }
  }
  reward = -static_cast<float>(min_dist);

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

  // Se estiver ruim, podemos tirar a media do loss
  // Dividir pelo tamanho das tentaivas

  // Backpropagation
  optimizer->zero_grad();
  loss.backward();
  optimizer->step();
}

void Smart::calculate_dist(uint_t time, uat::permit_public_status_fn status) {
  using namespace uat::permit_public_status;
  auto from = current_mission.from;
  auto to = current_mission.to;

  std::queue<Slot3d> q;
  auto short_path = from.shortest_path(to, rng());
  for (const auto&p : short_path) {
    auto idx = p.pos[0]*x + p.pos[1];
    full_dist[idx] = 0.0f;
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
      Slot3d new_slot{{static_cast<uint_t>(i), static_cast<uint_t>(j), 0}};
      if (std::holds_alternative<unavailable>(status(new_slot, time))) {
        full_dist[i*x+j] = 1e9f;
      }
    }
  }
}
