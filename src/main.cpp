#include "airspace3d.hpp"
#include "naive.hpp"

#include <cstdio>
#include <uat/simulation.hpp>
#include <cool/indices.hpp>
#include <cool/ccreate.hpp>
#include <random>

#include <CLI/CLI.hpp>

int main(int argc, char *argv[])
{
  using namespace uat;

  CLI::App app{"Simulate a First-price sealed-bid auction for airspace slots."};

  struct
  {
    uint_t max_time = 10;
    uint_t n_agents = 10;
    std::array<uint_t, 3> dimensions = {20, 20, 3};
    int seed = -1;

    std::string afilename;
    std::string tfilename;
    std::string pfilename;
  } opts;

  app.add_option("-t,--max-time", opts.max_time, "Factory maximum time");
  app.add_option("-n,--agents", opts.n_agents, "Number of agents generated each epoch");
  app.add_option("-d,--dimensions", opts.dimensions, "Airspace dimensions");
  app.add_option("-s,--seed", opts.seed, "Random seed (random_device if < 0)");

  app.add_option("-a,--agent-data", opts.afilename, "Save agent data to file");
  app.add_option("-p,--path-data", opts.pfilename, "Save agent path data to file");
  app.add_option("-o,--trade-data", opts.tfilename, "Save trades data to file");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  const auto open_file = [](const std::string& filename) -> std::FILE* {
    if (filename.empty())
      return nullptr;
    return filename == "-" ? stdout : std::fopen(filename.c_str(), "w");
  };

  const auto safe_close = [](std::FILE *fp) {
    if (fp && fp != stdout)
      std::fclose(fp);
  };

  const auto afile = cool::ccreate(open_file(opts.afilename), safe_close);
  if (afile)
    fmt::print(afile.get(), "Id,StartTime,Iterations,CongestionParam,FromX,FromY,FromZ,ToX,ToY,ToZ,Fundamental,Sigma,MinDistance,Distance\n");

  const auto pfile = cool::ccreate(open_file(opts.pfilename), safe_close);
  if (pfile)
    fmt::print(pfile.get(), "Id,X,Y,Z,Time\n");

  const auto tfile = cool::ccreate(open_file(opts.tfilename), safe_close);
  if (tfile)
    fmt::print(tfile.get(), "TransactionTime,From,To,X,Y,Z,Time,Value\n");

  auto factory = [&](uint_t t, const airspace& space, int seed) -> std::vector<agent> {
    if (t >= opts.max_time)
      return {};

    std::mt19937 rng(seed);

    std::vector<agent> result;
    result.reserve(opts.n_agents);
    for ([[maybe_unused]] const auto _ : cool::indices(opts.n_agents))
      result.push_back(Naive(space, rng(), afile.get(), pfile.get()));

    return result;
  };

  simulation_opts_t sim_opts = {
    /* .time_window = */ std::nullopt,
    /* .stop_criteria = */ uat::stop_criteria::no_agents_t{},
    // .trade_callback = TODO;
    nullptr,
    // .status_callback = TODO;
    nullptr,
  };

  simulate(factory, Airspace3D{opts.dimensions},
      opts.seed < 0 ? std::random_device{}() : opts.seed,
      sim_opts);
}
