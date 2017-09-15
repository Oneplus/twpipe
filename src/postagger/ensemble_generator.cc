#include "ensemble_generator.h"
#include "twpipe/logging.h"

namespace twpipe {

po::options_description EnsemblePostagDataGenerator::get_options() {
  po::options_description cmd("Ensemble data generate options.");
  cmd.add_options()
    ("ensemble-n-samples", po::value<unsigned>()->default_value(1), "the number of samples.")
    ;

  return cmd;
}

EnsemblePostagDataGenerator::EnsemblePostagDataGenerator(std::vector<PostagModel*>& engines,
                                                         const po::variables_map & conf) : engines(engines) {
  _INFO << "[twpipe|postag|ensemble_generator] number of ensembled parsers: " << engines.size();

  n_samples = conf["ensemble-n-samples"].as<unsigned>();
  _INFO << "[twpipe|postag|ensemble_generator] generate " << n_samples << " for each instance.";
}

void EnsemblePostagDataGenerator::generate(const std::vector<std::string>& words,
                                           const std::vector<std::string>& postags,
                                           std::vector<std::vector<float>>& prob) {
  unsigned n_engines = engines.size();
  dynet::ComputationGraph cg;

}

}
