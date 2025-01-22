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
  Smart(const Airspace3d&, int, size_t stateSize, size_t actionSize, double gamma = 0.99,
            double epsilon = 1.0, double epsilonMin = 0.01, double epsilonDecay = 0.995, long long replayMemorySize = 10000);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto on_sold(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

  std::vector<double> getAction(const std::vector<double>& state);

  void train();

  void storeExperience(const std::vector<double>& state, int action, double reward,
                        const std::vector<double>& nextState, bool done);

  void syncTargetNetwork();

  bool canAchieveMission(uat::uint_t);

private:
  Mission current_mission;
  uat::value_t spent = 0;
  std::mt19937 rng;
  std::unordered_set<uat::permit<Slot3d>> keep_, onsale_;

  std::uniform_real_distribution<> dist;
  uat::value_t fundamental_;
  std::uniform_real_distribution<> bid_value;

  torch::Device device;  // Simple initialization, OK in header
  std::shared_ptr<NeuralNetwork> qNetwork; // Default nullptr, OK in header
  std::shared_ptr<NeuralNetwork> targetNetwork; // Default nullptr, OK in header
  ReplayBuffer replayBuffer;
  std::unique_ptr<torch::optim::Adam> optimizer;

  double gamma;
  double epsilon;
  double epsilonMin;
  double epsilonDecay;
  size_t stateSize;
  size_t actionSize;

  int x;
  int y;

  std::vector<double> curr_state;
  std::vector<double> old_state;
  std::vector<double> last_action;
  std::vector<double> rewards;
  std::vector<double> log_probs;
};

static_assert(uat::agent_compatible<Smart>);
