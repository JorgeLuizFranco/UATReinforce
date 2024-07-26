#pragma once

#include <uat/airspace.hpp>
#include <uat/agent.hpp>
#include <uat/type.hpp>

// (First-price sealed-bid auction)
// TODO: explain t_heuristic
auto astar(const uat::region& from, const uat::region& to, uat::uint_t t0, uat::uint_t t_heuristic,
           uat::value_t fundamental,
           uat::value_t cost_per_time, uat::value_t turn_cost, uat::value_t climb_cost,
           uat::value_t max_cost,
           uat::permit_public_status_fn& status, int seed) -> std::vector<uat::permit>;
