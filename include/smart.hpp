#pragma once

#include <uat/type.hpp>
#include <uat/agent.hpp>

#include "airspace3d.hpp"

class Smart : public uat::agent_for<Slot3d>
{
public:
  Smart(const Airspace3D&, int);

  auto bid_phase(uat::uint_t, uat::bid_fn, uat::permit_public_status_fn, int) -> void override;

  auto ask_phase(uat::uint_t, uat::ask_fn, uat::permit_public_status_fn, int) -> void override;

  auto on_bought(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto on_sold(const Slot3d&, uat::uint_t, uat::value_t) -> void override;

  auto stop(uat::uint_t, int) -> bool override;

private:
  mission_t current_mission;
  uat::value_t spent = 0;
};

static_assert(uat::compatible_agent<Smart>);
