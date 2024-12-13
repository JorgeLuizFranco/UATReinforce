#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <limits>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "airspace3d.hpp"

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
  std::unordered_set<uat::permit<Slot3d>> keep_, onsale_;

  std::uniform_real_distribution<> dist;

  double alpha; // Learning rate
  double gamma; // Discount factor
  double epsilon;  // Exploration rate
};

static_assert(uat::agent_compatible<Smart>);
