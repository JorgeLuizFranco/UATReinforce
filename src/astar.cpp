#include "astar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <variant>

#include <cool/compose.hpp>

using namespace uat;

struct score_t {
  value_t g = std::numeric_limits<value_t>::infinity();
  value_t f = std::numeric_limits<value_t>::infinity();
};

template <typename T, typename Cmp>
class heap
{
public:
  heap(Cmp cmp) : cmp_(std::move(cmp)) {}

  auto push(T value)
  {
    for (auto& v : values_)
      if (v.first == value)
        v.second = false;
    values_.push_back({std::move(value), true});
    std::push_heap(values_.begin(), values_.end(), [&](const auto& x, const auto& y) {
      return cmp_(x.first, y.first);
    });
  }

  auto pop() -> std::pair<T, bool>
  {
    while (values_.size() > 1)
    {
      std::pop_heap(values_.begin(), values_.end(), [&](const auto& x, const auto& y) {
        return cmp_(x.first, y.first);
      });
      auto current = std::move(values_.back());
      values_.pop_back();
      if (current.second)
        return current;
    }
    auto value = std::move(values_.back());
    values_.pop_back();
    return value;
  }

  auto empty() const { return values_.empty(); }

private:
  std::vector<std::pair<T, bool>> values_;
  Cmp cmp_;
};

auto astar(const Slot2d& from, const Slot2d& to, uint_t t0, uint_t th, value_t bid,
           value_t icost, value_t turn_cost,
           value_t maxcost,
           permit_public_status_fn status, int seed) -> std::vector<permit<Slot2d>>
{
  using namespace uat::permit_public_status;
  if (std::holds_alternative<unavailable>(status(from, t0)))
    return {};

  assert(icost >= 0);
  assert(turn_cost >= 0);

  std::mt19937 gen(seed);

  // tries shortest path first
  {
    const auto candidate = [&]() -> std::vector<permit<Slot2d>> {
      auto path = from.shortest_path(to, gen());
      assert(path.size() == 0 || path.size() == from.distance(to) + 1);

      if (path.size() == 0)
        return {};

      std::vector<permit<Slot2d>> solution;
      for (const auto& slot : path)
      {
        if (std::holds_alternative<unavailable>(status(slot, t0)))
          return {};
        solution.push_back({slot, t0});
      }

      return solution;
    }();

    if (candidate.size() != 0)
      return candidate;
  }

  // tie-break trick: https://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#breaking-ties
  const auto trick = 1.0 + icost / from.distance(to);

  const auto h = [&](const Slot2d& s, uint_t t) -> value_t {
    const auto hash = std::hash<permit<Slot2d>>{}({s, t}) % 1000;
    const auto yatrick = 0.001 * hash / 1000.0;

    if (t > th) // I am sure that it will need to bid
      return s.heuristic_distance(to) * (icost + bid) * trick + yatrick;
    return s.heuristic_distance(to) * icost * trick + yatrick;
  };

  const auto cost = [&](const permit<Slot2d>& s, bool turn) {
    return std::visit(cool::compose{
      [&](unavailable) { return std::numeric_limits<value_t>::infinity(); },
      [&](owned) { return icost; },
      [&](available status) {
        return status.min_value > bid ?
          std::numeric_limits<value_t>::infinity() :
          icost + bid;
      }
    }, status(s.location, s.time)) + (turn ? turn_cost : 0.0);
  };

  std::unordered_map<permit<Slot2d>, permit<Slot2d>> came_from;
  std::unordered_map<permit<Slot2d>, score_t> score;

  score[{from, t0}] = {0, h(from, t0)};

  const auto cmp = [&score](const permit<Slot2d>& a, const permit<Slot2d>& b) {
    return score[a].f > score[b].f;
  };

  heap<permit<Slot2d>, decltype(cmp)> open(cmp);
  open.push({from, t0});

  const auto try_path = [&](const permit<Slot2d>& current, permit<Slot2d> next, bool turn) {
    const auto d = cost(next, turn);
    if (std::isinf(d))
      return;

    const auto tentative = score[current].g + d;
    const auto hnext = h(next.location, next.time);

    if (tentative < score[next].g && tentative + hnext < maxcost) {
      came_from.insert_or_assign(next, current);
      score[next] = {tentative, tentative + hnext};

      open.push(std::move(next));
    }
  };

  while (!open.empty())
  {
    const auto [current, valid]= open.pop();
    if (!valid)
      continue;

    if (current.location == to) {
      std::vector<permit<Slot2d>> path;
      path.push_back(current);

      while (path.back().location != from)
        path.push_back(came_from.at(path.back()));

      return path;
    }

    // XXX: should we keep forbiding staying still?
    // try_path(current, {current.location, current.time + 1});

    auto nei = current.location.neighbors();
    std::shuffle(nei.begin(), nei.end(), gen);
    for (auto nslot : nei)
    {
      const auto& before = current.location == from ? from : came_from.find(current)->second.location;
      const auto turn = current.location.turn(before, nslot);
      try_path(current, {std::move(nslot), current.time}, turn);
    }
  }

  return {};
}
