#include "noisify.h"
#include "logging.h"
#include "dynet/dynet.h"

po::options_description Noisifier::get_options() {
  po::options_description cmd("Noisify options");
  cmd.add_options()
    ("noisify_method", po::value<std::string>()->default_value("none"), "The type of noisifying method [none|singleton|word]")
    ("noisify_singleton_dropout_prob", po::value<float>()->default_value(0.2f), "The probability of dropping singleton, used in singleton mode.")
    ;
  return cmd;
}

Noisifier::Noisifier(const po::variables_map & conf, Corpus & c) : corpus(c) {
  std::string noisify_method_name = conf["noisify_method"].as<std::string>();
  if (noisify_method_name == "none") {
    noisify_method = kNone;
  } else if (noisify_method_name == "singleton") {
    noisify_method = kSingletonDroput;
  } else if (noisify_method_name == "word") {
    noisify_method = kWordDropout;
    singleton_dropout_prob = conf["noisify_singleton_dropout_prob"].as<float>();
  } else {
    _WARN << "Noisifier:: Unknown noisify method " << noisify_method_name << ", disable noisify.";
    noisify_method = kNone;
  }
  _INFO << "Noisifier:: method = " << noisify_method_name;
  unk = corpus.get_or_add_word(Corpus::UNK);
}

void Noisifier::noisify(InputUnits & units) const {
  if (noisify_method == kNone) {
    return;
  } else if (noisify_method == kSingletonDroput) {
    for (auto& u : units) {
      auto count = corpus.counter.find(u.wid);
      if (count != corpus.counter.end() &&
          count->second == 1 &&
          dynet::rand01() < singleton_dropout_prob) {
        u.wid = unk;
      }
    }
  } else {
    for (auto& u : units) {
      auto count = corpus.counter.find(u.wid);
      float prob = 0.25 / (0.25 + (count == corpus.counter.end() ? 0. : count->second));
      if (dynet::rand01() < prob) {
        u.wid = unk;
      }
    }
  }
}

void Noisifier::denoisify(InputUnits & units) const {
  for (auto& u : units) { u.wid = u.aux_wid; }
}
