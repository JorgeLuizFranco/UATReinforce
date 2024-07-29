#pragma once

#include <limits>
#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <unordered_set>
#include <cstdio>

#include "airspace3d.hpp"

// (First-price sealed-bid auction, same as the paper)
class Naive : public uat::agent<Slot3d>
{
public:
  Naive(const Airspace3D&, int, std::FILE*, std::FILE*);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

private:
  mission_t mission_;

  uat::value_t fundamental_;
  uat::value_t sigma_;

  uat::uint_t congestion_param_ = 1;
  uat::uint_t last_time_ = std::numeric_limits<uat::uint_t>::max();

  std::unordered_set<uat::permit<Slot3d>> keep_, onsale_;

  uat::uint_t niter_ = 0;

  std::FILE *agent_fp_, *path_fp_;
  bool ended = false;
};

static_assert(uat::agent_compatible<Naive>);
