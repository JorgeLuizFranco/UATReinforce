#include "airspace3d.hpp"

#include <cool/indices.hpp>
#include <fmt/core.h>

#include <algorithm>
#include <iterator>
#include <random>
#include <tuple>
#include <utility>

Airspace3D::Airspace3D(std::array<uint_t, 3> dim) : dim_{dim}
{
  assert(dim_[0] > 1);
  assert(dim_[1] > 1);
  assert(dim_[2] > 1);
}

auto Airspace3D::random_mission(int seed) const -> uat::mission_t
{
  std::mt19937 g(seed);

  std::array<std::uniform_int_distribution<uint_t>, 2> d{
    std::uniform_int_distribution<uint_t>{0u, dim_[0] - 1},
    std::uniform_int_distribution<uint_t>{0u, dim_[1] - 1},
  };

  const std::array<uint_t, 3> from = {d[0](g), d[1](g), 0};
  auto to = from;

  do {
    for (const auto i : {0u, 1u})
      to[i] = d[i](g);
  } while (from == to);

  return {Slot3d{from, dim_}, Slot3d{to, dim_}};
}

auto Airspace3D::dimensions() const -> std::array<uint_t, 3u> { return dim_; }

auto Slot3d::adjacent_regions() const -> std::vector<uat::region>
{
  std::vector<uat::region> nei;
  nei.reserve(6);

  if (pos[0] > 0) nei.push_back(Slot3d{{pos[0] - 1, pos[1], pos[2]}, dim});
  if (pos[0] < dim[0] - 1) nei.push_back(Slot3d{{pos[0] + 1, pos[1], pos[2]}, dim});
  if (pos[1] > 0) nei.push_back(Slot3d{{pos[0], pos[1] - 1, pos[2]}, dim});
  if (pos[1] < dim[1] - 1) nei.push_back(Slot3d{{pos[0], pos[1] + 1, pos[2]}, dim});
  if (pos[2] > 0) nei.push_back(Slot3d{{pos[0], pos[1], pos[2] - 1}, dim});
  if (pos[2] < dim[2] - 1) nei.push_back(Slot3d{{pos[0], pos[1], pos[2] + 1}, dim});

  return nei;
}

auto Slot3d::hash() const -> std::size_t
{
  return pos[0] * dim[1] * dim[2] + pos[1] * dim[2] + pos[2];
}


auto Slot3d::operator==(const Slot3d& other) const -> bool
{
  return pos == other.pos;
}

auto Slot3d::distance(const Slot3d& other) const -> uint_t
{
  constexpr auto diff = [](auto x, auto y) { return x > y ? x - y : y - x; };
  return
    diff(pos[0], other.pos[0]) +
    diff(pos[1], other.pos[1]) +
    diff(pos[2], other.pos[2]);
}

auto Slot3d::shortest_path(const Slot3d& to, int seed) const -> std::vector<uat::region>
{
  // prefers L instead of diagonal
  std::vector<uat::region> result;
  auto current = *this;

  result.push_back(current);

  std::mt19937 gen(seed);
  std::vector<uint_t> indexes = {0, 1, 2};
  std::shuffle(indexes.begin(), indexes.end(), gen);

  for (const auto i : indexes)
  {
    while (current.pos[i] != to.pos[i])
    {
      current.pos[i] += to.pos[i] > current.pos[i] ? +1 : -1;
      result.push_back(current);
    }
  }

  return result;
}

auto Slot3d::print(std::function<void(std::string_view, fmt::format_args)> f) const -> void
{
  f("{},{},{}", fmt::make_format_args(pos[0], pos[1], pos[2]));
}

auto Slot3d::turn(const Slot3d& before, const Slot3d& to) const -> bool
{
  const auto deltax = static_cast<long>(pos[0]) - static_cast<long>(before.pos[0]);
  const auto deltay = static_cast<long>(pos[1]) - static_cast<long>(before.pos[1]);

  assert(deltax >= -1 && deltax <= 1);
  assert(deltay >= -1 && deltay <= 1);

  if (deltax == 0 && pos[0] != to.pos[0])
    return true;

  if (deltay == 0 && pos[1] != to.pos[1])
    return true;

  if (pos[0] != to.pos[0] && pos[0] + deltax != to.pos[0])
    return true;

  if (pos[1] != to.pos[1] && pos[1] + deltay != to.pos[1])
    return true;

  return false;
}

auto Slot3d::climb(const Slot3d& to) const -> bool
{
  return to.pos[2] > pos[2];
}
