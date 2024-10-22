#include <cool/compose.hpp>
#include <cool/indices.hpp>
#include <jules/base/numeric.hpp>
#include <range/v3/view/transform.hpp>

#include <cstdio>
#include <iterator>
#include <cassert>
#include <limits>
#include <random>
#include <unordered_set>
#include <utility>
#include <variant>

#include "smart.hpp"

#include <fmt/core.h>

using namespace uat;

Smart::Smart(const Airspace3d& airspace, int seed)
  : current_mission(airspace.random_mission(seed)),
    QTable(airspace),
    rng(seed)
{
  std::mt19937 rng(seed);

  // Q learning variables
  alpha = 0.1;
  gamma = 0.9;
  epsilon = 0.1;
  // QTable = airspace;
  std::uniform_real_distribution<> dist;

  current_mission = airspace.random_mission(rng());

}

int Smart::choose_action(const Slot3d& state) {
    if (dist(rng) < epsilon) {
        // Explore: choose a random action
        return rand() % 2; // Assuming two actions: 0 and 1
    } else {
        // Exploit: choose the best action
        auto it = q_table.find(state);
        if (it != q_table.end()) {
            return std::max_element(it->second.begin(), it->second.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
        } else {
            return rand() % 2; // If state not found, choose random action
        }
    }
}

void Smart::update_q_table(const Slot3d& state, int action, double reward, const std::string& next_state) {
    double max_future_q = 0.0;
    auto it = q_table.find(next_state);
    if (it != q_table.end()) {
        max_future_q = std::max_element(it->second.begin(), it->second.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->second;
    }
    q_table[state][action] += alpha * (reward + gamma * max_future_q - q_table[state][action]);
}

auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  uat::uint_t target_time = time + 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {

    }
  }

}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{

}

auto Smart::on_bought(const Slot3d&, uat::uint_t, uat::value_t v) -> void
{
  spent += v;

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
  fmt::print(stderr, "Smart agent has stopped at time {}\n", t);
  return true;
}
