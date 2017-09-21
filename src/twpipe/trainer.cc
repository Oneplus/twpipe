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
    ("train-distill-postagger", po::value<bool>()->default_value(false), "train distilling parser.")
    ("train-parser", po::value<bool>()->default_value(false), "use to specify training the parser.")
    ("train-distill-parser", po::value<bool>()->default_value(false), "train distilling parser.")
    ("lambda", po::value<float>()->default_value(0.f), "the tense of l2")
    ("max-iter", po::value<unsigned>()->default_value(100), "the maximum number of training.")
    ("evaluate-stops", po::value<unsigned>()->default_value(0), "perform early stopping.")
    ("evaluate-skips", po::value<unsigned>()->default_value(0), "skip the first n evaluation.")
    ;
  return training_opts;
}

Trainer::Trainer(const po::variables_map & conf) {
  max_iter = conf["max-iter"].as<unsigned>();
  evaluate_stops = conf["evaluate-stops"].as<unsigned>();
  evaluate_skips = conf["evaluate-skips"].as<unsigned>();
  lambda_ = conf["lambda"].as<float>();
}

bool Trainer::need_evaluate(unsigned iter) {
  return iter > evaluate_skips;
}

bool Trainer::need_evaluate(unsigned iter, unsigned n_trained) {
  return ((iter > evaluate_skips) && (n_trained % evaluate_stops == 0));
}

}