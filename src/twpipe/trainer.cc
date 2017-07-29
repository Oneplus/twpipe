#include "trainer.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/assert.hpp>

namespace twpipe {

po::options_description Trainer::get_options() {
  po::options_description training_opts("Training options");
  training_opts.add_options()
    ("heldout", po::value<std::string>(), "heldout data file")
    ("train-tokenizer", po::value<bool>()->default_value(false), "use to specify training the tokenizer.")
    ("train-postagger", po::value<bool>()->default_value(false), "use to specify training the postagger.")
    ("train-parser", po::value<bool>()->default_value(false), "use to specify training the parser.")
    ("max-iter", po::value<unsigned>()->default_value(100), "the maximum number of training.")
    ("early-stop", po::value<bool>()->default_value(false), "perform early stopping.")
    ;
  return training_opts;
}

Trainer::Trainer(const po::variables_map & conf) {
  max_iter = conf["max-iter"].as<unsigned>();
  early_stop = conf["early-stop"].as<bool>();
}

}