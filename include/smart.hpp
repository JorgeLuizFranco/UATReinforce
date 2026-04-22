#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <cstdio>
#include <random>
#include <vector>
#include <memory>

#include "airspace2d.hpp"

#include "network.hpp"

class Smart : public uat::agent<Slot2d>
{
public:
  Smart(const Airspace2d&, int, size_t stateSize, size_t actionSize, float learning_rate = 0.001);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot2d&, uat::uint_t, uat::value_t) -> void override;

  auto on_sold(const Slot2d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

  std::vector<float> getAction(const std::vector<float>& state);

  bool canAchieveMission(uat::uint_t);

  void clear_episode();

private:
  Mission current_mission;
  uat::value_t spent = 0;
  std::mt19937 rng;
  std::unordered_set<uat::permit<Slot2d>> keep_, onsale_;
  Airspace2d space;

  torch::Device device;
  std::shared_ptr<NeuralNetwork> qNetwork;
  std::unique_ptr<torch::optim::Adam> optimizer;

  torch::Tensor compute_returns();
  std::tuple<std::string, float, bool> can_achieve_mission(uint_t t);
  std::vector<float> calculate_dist(uint_t time, uat::permit_public_status_fn status);
  void clean_states(uint_t t);
  void back_propagation();

  bool can_finish_mission(uint_t curr_time, uat::permit_public_status_fn status);
  std::vector<float> stack_states(const std::vector<const std::vector<float>*>& states);

  // Neural network params
  float learning_rate;
  float gamma;
  size_t stateSize;
  size_t actionSize;

  // Airspace dimensions
  int x;
  int y;

  // Reward function parameters
  float faturamento_bruto;
  std::vector<float> rewards;
  std::vector<torch::Tensor> log_probs;

  // Time controlling variables
  int episode_counter = 0;
  int time_steps = 5;
  uat::uint_t curr_time;

  // Agent states
  std::vector<float> curr_state;
  std::vector<float> old_state;
};

static_assert(uat::agent_compatible<Smart>);
