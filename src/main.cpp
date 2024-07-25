#include <uat/simulation.hpp>
#include <CLI/CLI.hpp>

#include <random>

struct Airspace {
  auto random_mission(int) const -> uat::mission_t { return {}; }
  auto iterate(uat::region_fn) const -> void {}
};

int main(int argc, char *argv[])
{
  using namespace uat;

  CLI::App app{"A Reinforcement Learning Framework for Urban Airspace Tradable Permit Model."};

  struct
  {
    int seed = -1;
  } opts;

  app.add_option("-s,--seed", opts.seed, "Random seed (random_device if < 0)");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  auto factory = [](uint_t, const airspace&, int) -> std::vector<agent> {
    return {};
  };

  simulate(factory, Airspace{},
      opts.seed < 0 ? std::random_device{}() : opts.seed);
}
