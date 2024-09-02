#pragma once

#include <fmt/core.h>
#include <type_traits>
#include <uat/permit.hpp>

using uint_t = uat::uint_t;

struct Slot3d
{
  std::array<uint_t, 3> pos, dim;

  auto neighbors() const -> std::vector<Slot3d>;

  auto hash() const -> std::size_t;

  auto operator==(const Slot3d&) const -> bool;
  auto operator!=(const Slot3d&) const -> bool;

  auto distance(const Slot3d&) const -> uint_t;

  auto heuristic_distance(const Slot3d&) const -> double;

  auto shortest_path(const Slot3d&, int) const -> std::vector<Slot3d>;

  auto turn(const Slot3d& before, const Slot3d& to) const -> bool;

  auto climb(const Slot3d& to) const -> bool;
};

template <> struct std::hash<Slot3d>
{
  auto operator()(const Slot3d& s) const -> std::size_t;
};

struct Mission {
  Slot3d from, to;
  auto distance() const -> uint_t;
};

class Airspace3d
{
public:
  explicit Airspace3d(std::array<uint_t, 3>);

  auto random_mission(int) const -> Mission;

  auto dimensions() const -> std::array<uint_t, 3u>;

private:
  std::array<uint_t, 3> dim_;
};
