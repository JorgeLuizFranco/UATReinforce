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
    rng(seed)
{
  std::mt19937 rng(seed);

  // Q learning variables
  alpha = 0.1;
  gamma = 0.9;
  epsilon = 0.1;

  std::uniform_real_distribution<> dist;
  current_mission = airspace.random_mission(rng());
}


auto Smart::bid_phase(uat::uint_t time, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  uat::uint_t target_time = time + 1;
}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{

}

auto Smart::on_bought(const Slot3d& location, uat::uint_t time, uat::value_t v) -> void
{
  spent += v;

  keep_.insert({location, time});
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
