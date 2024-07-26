#pragma once

#include <fmt/core.h>
#include <type_traits>
#include <uat/type.hpp>
#include <uat/airspace.hpp>
#include <uat/permit.hpp>

using uint_t = uat::uint_t;

class Airspace3D
{
public:
  explicit Airspace3D(std::array<uint_t, 3>);

  auto random_mission(int) const -> uat::mission_t;

  auto dimensions() const -> std::array<uint_t, 3u>;

  auto iterate(uat::region_fn) const -> void { /* TODO */ }

private:
  std::array<uint_t, 3> dim_;
};

struct Slot3d
{
  std::array<uint_t, 3> pos, dim;

  auto adjacent_regions() const -> std::vector<uat::region>;

  auto hash() const -> std::size_t;

  auto operator==(const Slot3d&) const -> bool;

  auto distance(const Slot3d&) const -> uint_t;

  auto shortest_path(const Slot3d&, int) const -> std::vector<uat::region>;

  auto print(std::function<void(std::string_view, fmt::format_args)>) const -> void;

  auto turn(const Slot3d& before, const Slot3d& to) const -> bool;

  auto climb(const Slot3d& to) const -> bool;
};
