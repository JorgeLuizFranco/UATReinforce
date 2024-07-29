#include "naive.hpp"

#include "astar.hpp"

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

using namespace uat;

Naive::Naive(const Airspace3D& space, int seed, std::FILE* agent_fp, std::FILE* path_fp) :
  agent_fp_{agent_fp}, path_fp_{path_fp}
{
  std::mt19937 rng(seed);

  std::uniform_real_distribution<value_t> f{50.0, 150.0};
  std::uniform_real_distribution<value_t> s{0.0, 0.2};

  fundamental_ = f(rng);
  sigma_ = s(rng);

  mission_ = space.random_mission(rng());
}

auto Naive::bid_phase(uint_t t, uat::bid_fn bid, uat::permit_public_status_fn status, int seed) -> void
{
  ++niter_;

  std::mt19937 rng(seed);
  std::normal_distribution<value_t> dist{0.0, sigma_ * fundamental_};

  onsale_ = std::exchange(keep_, {});
  const auto t_heuristic =
    jules::max(onsale_ | ranges::views::transform([](const permit<Slot3d>& s) { return s.time(); }));

  // check previous path
  if (last_time_ != std::numeric_limits<uat::uint_t>::max())
  {
    auto path = astar(mission_.from, mission_.to, last_time_,
        t_heuristic,
        std::numeric_limits<value_t>::infinity(), 1.0, 0.1, 0.2,
        std::numeric_limits<value_t>::infinity(), status, rng());

    // mission completed
    if (path.size() != 0) {
      for (auto& position : path) {
        onsale_.erase(position);
        keep_.insert(std::move(position));
      }
      ended = true;
      return;
    }
  }

  const auto path = [&]{
    uint_t start = 1u;
    std::vector<uint_t> tries;

    while (true)
    {
      tries.clear();
      tries.reserve(congestion_param_ - start + 1);

      ranges::copy(ranges::views::closed_iota(start, congestion_param_), ranges::back_inserter(tries));
      ranges::shuffle(tries, rng);

      for (const auto wait : tries)
      {
        last_time_ = t + wait;
        auto p = astar(mission_.from, mission_.to, last_time_, t_heuristic,
            fundamental_, 1.0, 0.1, 0.2, std::numeric_limits<value_t>::infinity(), status, rng());

        if (p.size() > 0) {
          congestion_param_ *= 2;
          return p;
        }
      }

      start = congestion_param_ + 1;
      congestion_param_ *= 2;
    }
  }();

  for (const auto& [region, t] : path) // see: https://stackoverflow.com/questions/46114214/lambda-implicit-capture-fails-with-variable-declared-from-structured-binding
  {
    std::visit(cool::compose{
      [](uat::permit_public_status::unavailable) { assert(false); },
      [&, region = region, t = t](uat::permit_public_status::owned) {
        onsale_.erase(uat::permit<Slot3d>(region, t));
        keep_.emplace(std::move(region), t);
      },
      [&, region = region, t = t](uat::permit_public_status::available) {
        bid(std::move(region), t, fundamental_ - std::abs(dist(rng)));
      },
    }, status(region, t));
  }

  ended = keep_.size() == path.size(); // false if there are missing waypoints
}

auto Naive::ask_phase(uint_t, uat::ask_fn ask, uat::permit_public_status_fn , int) -> void
{
  for (const auto& position : onsale_)
    ask(position.location(), position.time(), 0.0);
  onsale_.clear();
}

auto Naive::on_bought(const Slot3d& s, uint_t t, value_t) -> void
{
  keep_.insert({s, t});
}

auto Naive::stop(uint_t, int) -> bool
{
  return ended;

  // XXX not working anymore, agents shouldn't be able to know their id.
  // Private status must be used instead
  // if (path_fp_)
  // {
  //   for (const auto& [region, t] : keep_)
  //     fmt::print(path_fp_, "{},{},{}\n",
  //         id, uat::region(region), t);
  // }

  // if (agent_fp_)
  // {
  //   fmt::print(agent_fp_, "{},{},{},{},{},{},{},{},{},{}\n",
  //       id, t + 1 - niter_, niter_, congestion_param_,
  //       uat::region(mission_.from), uat::region(mission_.to),
  //       fundamental_, sigma_,
  //       mission_.length(), keep_.size() - 1.0);
  // }
}

