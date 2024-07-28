#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include "airspace3d.hpp"

class Smart
{
public:
  Smart(const Airspace3D&, int);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void;

  auto on_bought(const uat::region&, uat::uint_t, uat::value_t) -> void;

  auto on_sold(const uat::region&, uat::uint_t, uat::value_t) -> void;

  auto stop(uat::uint_t, int) -> bool;

private:
  mission_t current_mission;
  uat::value_t spent = 0;
};

