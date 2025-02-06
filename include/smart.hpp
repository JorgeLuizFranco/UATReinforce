#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <limits>
#include <cstdio>
#include <map>
#include <random>
#include <vector>
#include <memory>

#include "airspace3d.hpp"

#include "network.hpp"

class Smart : public uat::agent<Slot3d>
{
public:
  Smart(const Airspace3d&, int, size_t stateSize, size_t actionSize, float learning_rate = 0.001);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto on_sold(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

  std::vector<float> getAction(const std::vector<float>& state);

  bool canAchieveMission(uat::uint_t);

  void clear_episode();

private:
  Mission current_mission;
  uat::value_t spent = 0;
  std::mt19937 rng;
  std::unordered_set<uat::permit<Slot3d>> keep_, onsale_;
  Airspace3d space;

  torch::Device device;  // Simple initialization, OK in header
  std::shared_ptr<NeuralNetwork> qNetwork; // Default nullptr, OK in header
  std::unique_ptr<torch::optim::Adam> optimizer;

  torch::Tensor compute_returns();
  std::tuple<std::string, float, bool> can_achieve_mission(uint_t t);
  void calculate_dist(uint_t time, uat::permit_public_status_fn status);
  void clean_states(uint_t t);
  void back_propagation();

  // Neural network params
  float learning_rate;
  float gamma;
  size_t stateSize;
  size_t actionSize;

  // Airspace dimensions
  int x;
  int y;

  std::vector<float> curr_state;
  std::vector<float> old_state;
  std::vector<float> last_action;
  std::vector<float> rewards;
  std::vector<float> full_dist;
  std::vector<torch::Tensor> log_probs;

  uat::uint_t curr_time;
};

static_assert(uat::agent_compatible<Smart>);
