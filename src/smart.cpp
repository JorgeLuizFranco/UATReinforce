#include "smart.hpp"

#include <fmt/core.h>
#include <uat/airspace.hpp>

Smart::Smart(const uat::airspace& airspace, int seed)
  : current_mission(airspace.random_mission(seed))
{
}

auto Smart::bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void
{
}

auto Smart::ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void
{

}

auto Smart::on_bought(const uat::region&, uat::uint_t, uat::value_t v) -> void
{
  spent += v;

  // check whether mission has been completed
  // then starts a new mission
  // mission_ = space.random_mission(rng());
}

auto Smart::on_sold(const uat::region&, uat::uint_t, uat::value_t v) -> void
{
  spent -= v;
}

auto Smart::stop(uat::uint_t t, int) -> bool
{
  fmt::print(stderr, "Smart agent has stopped at time {}\n", t);
  return true;
}
