#pragma once

#include <fmt/core.h>
#include <type_traits>
#include <uat/type.hpp>
#include <uat/permit.hpp>

using uint_t = uat::uint_t;

struct Slot3d
{
  std::array<uint_t, 3> pos, dim;

  auto adjacent_regions() const -> std::vector<Slot3d>;

  auto hash() const -> std::size_t;

  auto operator==(const Slot3d&) const -> bool;
  auto operator!=(const Slot3d& other) const -> bool { return !(*this == other); }

  auto distance(const Slot3d&) const -> uint_t;

  auto heuristic_distance(const Slot3d&) const -> double;

  auto shortest_path(const Slot3d&, int) const -> std::vector<Slot3d>;

  auto print(std::function<void(std::string_view, fmt::format_args)>) const -> void;

  auto turn(const Slot3d& before, const Slot3d& to) const -> bool;

  auto climb(const Slot3d& to) const -> bool;
};

auto operator<<(std::ostream&, const Slot3d&) -> std::ostream&;

namespace std {
template <>
struct hash<Slot3d> {
  auto operator()(const Slot3d& s) const -> std::size_t { return s.hash(); }
};
} // namespace std

static_assert(uat::region_compatible<Slot3d>);

struct mission_t {
  Slot3d from;
  Slot3d to;
  auto length() const { return from.distance(to); }
};

class Airspace3D
{
public:
  explicit Airspace3D(std::array<uint_t, 3>);

  auto random_mission(int) const -> mission_t;

  auto dimensions() const -> std::array<uint_t, 3u>;

private:
  std::array<uint_t, 3> dim_;
};
