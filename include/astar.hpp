#pragma once

#include <uat/agent.hpp>
#include <uat/type.hpp>

#include "airspace3d.hpp"

// (First-price sealed-bid auction)
// TODO: explain t_heuristic
auto astar(const Slot3d& from, const Slot3d& to, uat::uint_t t0, uat::uint_t t_heuristic,
           uat::value_t fundamental,
           uat::value_t cost_per_time, uat::value_t turn_cost, uat::value_t climb_cost,
           uat::value_t max_cost,
           uat::permit_public_status_fn& status, int seed) -> std::vector<uat::permit<Slot3d>>;
