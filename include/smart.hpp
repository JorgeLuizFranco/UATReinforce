#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <limits>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "airspace3d.hpp"

typedef struct {
  std::vector<std::vector<std::vector<int>>> pos_grid;
} State;

typedef struct {
  std::vector<std::vector<std::vector<double>>> bid_grid;
} Action;

class Smart : public uat::agent<Slot3d>
{
public:
  Smart(const Airspace3d&, int);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto on_sold(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

private:
  Mission current_mission;
  uat::value_t spent = 0;
  std::mt19937 rng;

  // 3d matrix representing the states (locations where the smart agent has permissions)
  // and actions (the bid for every position in space)
  State states;
  Action actions;

  std::map<std::pair<State, Action>, double> qtable;
  std::uniform_real_distribution<> dist;
  double alpha; // Learning rate
  double gamma; // Discount factor
  double epsilon;  // Exploration rate

  int choose_action(const Slot3d& state);
  void update_q_table(const Slot3d& state, int action, double reward, const std::string& next_state);
  double get_reward(const Airspace3d& state);
};

static_assert(uat::agent_compatible<Smart>);
