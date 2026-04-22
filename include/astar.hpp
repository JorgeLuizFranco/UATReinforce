#pragma once

#include <uat/agent.hpp>

#include "airspace2d.hpp"

// (First-price sealed-bid auction)
// TODO: explain t_heuristic
auto astar(const Slot2d& from, const Slot2d& to, uat::uint_t t0, uat::uint_t t_heuristic,
           uat::value_t fundamental,
           uat::value_t cost_per_time, uat::value_t turn_cost,
           uat::value_t max_cost,
           uat::permit_public_status_fn status, int seed) -> std::vector<uat::permit<Slot2d>>;
