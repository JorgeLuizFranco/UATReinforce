#include "airspace2d.hpp"

#include <cool/indices.hpp>
#include <fmt/core.h>

#include <algorithm>
#include <random>

Airspace2d::Airspace2d(std::array<uint_t, 2> dim) : dim_{dim}
{
  assert(dim_[0] > 1);
  assert(dim_[1] > 1);
}

auto Airspace2d::random_mission(int seed) const -> Mission
{
  std::mt19937 g(seed);

  std::array<std::uniform_int_distribution<uint_t>, 2> d{
    std::uniform_int_distribution<uint_t>{0u, dim_[0] - 1},
    std::uniform_int_distribution<uint_t>{0u, dim_[1] - 1},
  };

  const std::array<uint_t, 2> from = {d[0](g), d[1](g)};
  auto to = from;

  do {
    for (const auto i : {0u, 1u})
      to[i] = d[i](g);
  } while (from == to);

  return {Slot2d{from, dim_}, Slot2d{to, dim_}};
}

auto Airspace2d::dimensions() const -> std::array<uint_t, 2u> { return dim_; }

auto Slot2d::neighbors() const -> std::vector<Slot2d>
{
  std::vector<Slot2d> nei;
  nei.reserve(6);

  if (pos[0] > 0) nei.push_back(Slot2d{{pos[0] - 1, pos[1]}, dim});
  if (pos[0] < dim[0] - 1) nei.push_back(Slot2d{{pos[0] + 1, pos[1]}, dim});
  if (pos[1] > 0) nei.push_back(Slot2d{{pos[0], pos[1] - 1}, dim});
  if (pos[1] < dim[1] - 1) nei.push_back(Slot2d{{pos[0], pos[1] + 1}, dim});

  return nei;
}

auto Slot2d::hash() const -> std::size_t
{
  assert(pos[0] < dim[0]);
  assert(pos[1] < dim[1]);
  return pos[0] * dim[1] + pos[1];
}


auto Slot2d::operator==(const Slot2d& other) const -> bool
{
  return pos == other.pos;
}

auto Slot2d::operator!=(const Slot2d& other) const -> bool
{
  return pos != other.pos;
}

auto Slot2d::distance(const Slot2d& other) const -> uint_t
{
  constexpr auto diff = [](auto x, auto y) { return x > y ? x - y : y - x; };
  return
    diff(pos[0], other.pos[0]) +
    diff(pos[1], other.pos[1]);
}

auto Slot2d::heuristic_distance(const Slot2d& other) const -> double
{
  return static_cast<double>(distance(other));
}

auto Slot2d::shortest_path(const Slot2d& to, int seed) const -> std::vector<Slot2d>
{
  // prefers L instead of diagonal (more time moving straight)
  std::vector<Slot2d> result;
  auto current = *this;

  result.push_back(current);

  std::mt19937 gen(seed);
  std::vector<uint_t> indexes = {0, 1};
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

auto Slot2d::turn(const Slot2d& before, const Slot2d& to) const -> bool
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

auto std::hash<Slot2d>::operator()(const Slot2d& s) const -> std::size_t
{
  return s.hash();
}

auto Mission::distance() const -> uint_t
{
  return from.distance(to);
}

auto Airspace2d::to_vector() const -> std::vector<uint_t>
{
  return {dim_[0], dim_[1]};
}
