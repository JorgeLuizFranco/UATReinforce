#pragma once

#include <limits>
#include <uat/type.hpp>
#include <uat/agent.hpp>

#include <unordered_set>
#include <cstdio>

#include "airspace3d.hpp"

// (First-price sealed-bid auction, same as the paper)
class Naive
{
public:
  Naive(const Airspace3D&, int, std::FILE*, std::FILE*);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void;

  auto on_bought(const uat::region&, uat::uint_t, uat::value_t) -> void;

  auto stop(uat::uint_t, uat::uint_t) -> bool;

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
