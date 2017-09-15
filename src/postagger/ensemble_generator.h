#ifndef __TWPIPE_POSTAGGER_ENSEMBLE_GENERATOR_H__
#define __TWPIPE_POSTAGGER_ENSEMBLE_GENERATOR_H__

#include <vector>
#include <boost/program_options.hpp>
#include "postag_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct EnsemblePostagDataGenerator {
  unsigned n_samples;
  std::vector<PostagModel *>& engines;

  static po::options_description get_options();

  EnsemblePostagDataGenerator(std::vector<PostagModel *>& engines,
                              const po::variables_map & conf);

  void generate(const std::vector<std::string> & words,
                const std::vector<std::string> & postags,
                std::vector<std::vector<float>> & prob);

};

}

#endif  //  end for __TWPIPE_POSTAGGER_ENSEMBLE_GENERATOR_H__