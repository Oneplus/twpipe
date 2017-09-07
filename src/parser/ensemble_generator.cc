#include "ensemble_generator.h"
#include "twpipe/logging.h"

namespace twpipe {

po::options_description EnsembleDataGenerator::get_options() {
  po::options_description cmd("Ensemble data generate options.");
  cmd.add_options()
    ("ensemble-method", po::value<std::string>()->default_value("prob"), "ensemble methods [prob|logits_mean|logits_sum]")
    ("ensemble-n-samples", po::value<unsigned>()->default_value(1), "the number of samples.")
    ("ensemble-rollin", po::value<std::string>()->default_value("expert"), "the rollin-method [egreedy|boltzmann]")
    ("ensemble-egreedy-epsilon", po::value<float>()->default_value(0.1f), "the epsilon of epsilon-greedy policy.")
    ("ensemble-boltzmann-temperature", po::value<float>()->default_value(1.f), "the temperature of epsilon-greedy policy.")
    ;

  return cmd;
}

EnsembleDataGenerator::EnsembleDataGenerator(std::vector<ParseModel*>& engines,
                                             const po::variables_map & conf) : engines(engines) {
  _INFO << "[twpipe|parser|ensemble_generator] number of ensembled parsers: " << engines.size();

  std::string ensemble_method_name = conf["ensemble-method"].as<std::string>();
  if (ensemble_method_name == "prob") {
    ensemble_method = kProbability;
  } else if (ensemble_method_name == "logits_mean") {
    ensemble_method = kLogitsMean;
  } else if (ensemble_method_name == "logits_sum") {
    ensemble_method = kLogitsSum;
  } else {
    _ERROR << "[twpipe|parser|ensemble_generator] unknown ensemble method: " << ensemble_method_name;
    exit(1);
  }
  _INFO << "[twpipe|parser|ensemble_generator] ensemble method: " << ensemble_method_name;

  n_samples = conf["ensemble-n-samples"].as<unsigned>();
  _INFO << "[twpipe|parser|ensemble_generator] generate " << n_samples << " for each instance.";

  std::string rollin_name = conf["ensemble-rollin"].as<std::string>();
  if (rollin_name == "egreedy") {
    rollin_policy = kEpsilonGreedy;
    epsilon = conf["ensemble-egreedy-epsilon"].as<float>();
    _INFO << "[twpipe|parser|ensemble_generator] roll-in policy: " << rollin_name;
    _INFO << "[twpipe|parser|ensemble_generator] epsilon: " << epsilon;
  } else if (rollin_name == "boltzmann") {
    rollin_policy = kBoltzmann;
    temperature = conf["ensemble-boltzmann-temperature"].as<float>();
    _INFO << "[twpipe|parser|ensemble_generator] roll-in policy: " << rollin_name;
    _INFO << "[twpipe|parser|ensemble_generator] temperature: " << temperature;
  } else {
    _ERROR << "[twpipe|parser|ensemble_generator] unknown roll-in policy: " << rollin_name;
    exit(1);
  }
}

void EnsembleDataGenerator::generate(const std::vector<std::string>& words,
                                     const std::vector<std::string>& postags, 
                                     std::vector<unsigned>& actions,
                                     std::vector<std::vector<float>>& prob) {
}

}
