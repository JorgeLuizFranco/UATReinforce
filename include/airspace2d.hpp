#pragma once

#include <fmt/core.h>
#include <uat/permit.hpp>

using uint_t = uat::uint_t;

struct Slot2d
{
  std::array<uint_t, 2> pos, dim;

  auto neighbors() const -> std::vector<Slot2d>;

  auto hash() const -> std::size_t;

  auto operator==(const Slot2d&) const -> bool;
  auto operator!=(const Slot2d&) const -> bool;

  auto distance(const Slot2d&) const -> uint_t;

  auto heuristic_distance(const Slot2d&) const -> double;

  auto shortest_path(const Slot2d&, int) const -> std::vector<Slot2d>;

  auto turn(const Slot2d& before, const Slot2d& to) const -> bool;
};

template <> struct std::hash<Slot2d>
{
  auto operator()(const Slot2d& s) const -> std::size_t;
};

struct Mission {
  Slot2d from, to;
  auto distance() const -> uint_t;
};

class Airspace2d
{
public:
  explicit Airspace2d(std::array<uint_t, 2>);

  auto random_mission(int) const -> Mission;

  auto dimensions() const -> std::array<uint_t, 2u>;

  auto to_vector() const -> std::vector<uint_t>;
private:
  std::array<uint_t, 2> dim_;
};
