#ifndef __TWPIPE_ENSEMBLE_GENERATOR_H__
#define __TWPIPE_ENSEMBLE_GENERATOR_H__

#include <vector>
#include <boost/program_options.hpp>
#include "parse_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct EnsembleDataGenerator {
  enum ENSEMBLE_METHOD_TYPE { kProbability, kLogitsMean, kLogitsSum };
  enum ROLLIN_POLICY_TYPE { kEpsilonGreedy, kBoltzmann };

  ENSEMBLE_METHOD_TYPE ensemble_method;
  ROLLIN_POLICY_TYPE rollin_policy;
  unsigned n_samples;
  float epsilon;
  float temperature;
  
  std::vector<ParseModel *>& engines;

  static po::options_description get_options();

  EnsembleDataGenerator(std::vector<ParseModel *>& engines,
                        const po::variables_map & conf);

  void generate(const std::vector<std::string> & words,
                const std::vector<std::string> & postags,
                std::vector<unsigned> & actions,
                std::vector<std::vector<float>> & prob);
};

}

#endif  //  end for __TWPIPE_ENSEMBLE_GENERATOR_H__