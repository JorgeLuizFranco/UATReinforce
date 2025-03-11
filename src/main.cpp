#include "airspace2d.hpp"
#include "naive.hpp"
#include "smart.hpp"

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
    uint_t max_time = 100000;
    uint_t n_agents = 10;
    std::array<uint_t, 2> dimensions = {15, 15};
    int seed = -1;
    bool dummy = false;

    std::string afilename;
    std::string tfilename;
    std::string pfilename;
  } opts;

  app.add_option("-t,--max-time", opts.max_time, "Factory maximum time");
  app.add_option("-n,--agents", opts.n_agents, "Number of agents generated each epoch");
  app.add_option("-d,--dimensions", opts.dimensions, "Airspace dimensions");
  app.add_option("-s,--seed", opts.seed, "Random seed (random_device if < 0)");
  app.add_option("--dummy", opts.dummy, "Do not instantiate smart agent");

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

  Airspace2d space{opts.dimensions};

  auto factory = [&, id = uint_t{0}](uint_t t, int seed) mutable -> std::vector<any_agent> {
    if (t >= opts.max_time)
      return {};

    std::mt19937 rng(seed);

    std::vector<any_agent> result;
    result.reserve(opts.n_agents + (t == 0 ? 1 : 0));

    if (t == 0 and not opts.dummy)
      result.push_back(Smart(space, 42, 225, 225));

    for ([[maybe_unused]] const auto _ : cool::indices(opts.n_agents)){
      // fmt::print("Factory: Creating agent {} at time {}\n", id, t);
      result.push_back(Naive(id++, space, rng(), afile.get(), pfile.get()));
    }

    // fmt::print("Factory: Returning {} agents at time {}\n", result.size(), t);

    return result;
  };

  simulate<Slot2d>({
    .factory = std::move(factory),
    .time_window = std::nullopt,
    .stop_criterion = stop_criterion::time_threshold_t{opts.max_time},
    .trade_callback = tfile ? [&](trade_info_t<Slot2d> trade) {
      if (trade.from != no_owner)
        fmt::print(tfile.get(), "{},{},{},{},{},{},{},{}\n",
                   trade.transaction_time, trade.from, trade.to,
                   trade.location.pos[0], trade.location.pos[1], trade.location.pos[2],
                   trade.time, trade.value);
      else
        fmt::print(tfile.get(), "{},NA,{},{},{},{},{},{}\n",
                   trade.transaction_time, trade.to,
                   trade.location.pos[0], trade.location.pos[1], trade.location.pos[2],
                   trade.time, trade.value);
    } : std::function<void(trade_info_t<Slot2d>)>(),
    .simulation_callback = [&](uint_t iteration, const agents_private_status_t& status,
                             permit_private_status_fn) -> void {
      fmt::print("Iteration: {} | Active Agents: {}\n", iteration, status.active_count());
    },
    .seed = opts.seed < 0 ? std::random_device{}() : opts.seed,
  });
}
