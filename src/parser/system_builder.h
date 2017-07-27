#ifndef SYSTEM_BUILDER_H
#define SYSTEM_BUILDER_H

#include <boost/program_options.hpp>
#include "system.h"
#include "corpus.h"

namespace po = boost::program_options;

struct TransitionSystemBuilder {
  static po::options_description get_options();
  Corpus& corpus;

  TransitionSystemBuilder(Corpus& corpus);

  TransitionSystem* build(const po::variables_map & conf);
  static bool allow_nonprojective(const po::variables_map & conf);
};

#endif  //  end for SYSTEM_BUILDER_H