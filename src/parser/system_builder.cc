#include "system_builder.h"
#include "logging.h"
#include "arcstd.h"
#include "arceager.h"
#include "archybrid.h"
#include "swap.h"

po::options_description TransitionSystemBuilder::get_options() {
  po::options_description cmd("Transition system options");
  cmd.add_options()
    ("system", po::value<std::string>()->default_value("arcstd"), "The transition system [arcstd, arceager, archybrid, swap].")
    ;

  return cmd;
}

TransitionSystemBuilder::TransitionSystemBuilder(Corpus & c) : corpus(c) {
}

TransitionSystem * TransitionSystemBuilder::build(const po::variables_map & conf) {
  TransitionSystem * sys;
  std::string system_name = conf["system"].as<std::string>();
  if (system_name == "arcstd") {
    sys = new ArcStandard(corpus.deprel_map);
  } else if (system_name == "arceager") {
    sys = new ArcEager(corpus.deprel_map);
  } else if (system_name == "archybrid") {
    sys = new ArcHybrid(corpus.deprel_map);
  } else if (system_name == "swap") {
    sys = new Swap(corpus.deprel_map);
  } else {
    _ERROR << "SysBuilder:: Unknown transition system: " << system_name;
    exit(1);
  }
  _INFO << "SysBuilder:: transition system: " << system_name;
  return sys;
}

bool TransitionSystemBuilder::allow_nonprojective(const po::variables_map & conf) {
  std::string system_name = conf["system"].as<std::string>();
  return system_name == "swap";
}
