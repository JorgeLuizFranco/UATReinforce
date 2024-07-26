#include "astar.hpp"

auto astar(const uat::region& from, const uat::region& to, uat::uint_t t0, uat::uint_t t_heuristic,
           uat::value_t fundamental,
           uat::value_t cost_per_time, uat::value_t turn_cost, uat::value_t climb_cost,
           uat::value_t max_cost,
           uat::permit_public_status_fn& status, int seed) -> std::vector<uat::permit>
{
  (void)from;
  (void)to;
  (void)t0;
  (void)t_heuristic;
  (void)fundamental;
  (void)cost_per_time;
  (void)turn_cost;
  (void)climb_cost;
  (void)max_cost;
  (void)status;
  (void)seed;
  return {};
}
