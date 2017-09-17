#include "ensemble_generator.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/math.h"

namespace twpipe {

po::options_description EnsemblePostagDataGenerator::get_options() {
  po::options_description cmd("Ensemble data generate options.");
  cmd.add_options()
    ("ensemble-n-samples", po::value<unsigned>()->default_value(1), "the number of samples.")
    ("ensemble-rollin", po::value<std::string>()->default_value("predict"), "the rollin-method [expert|predict]")
    ("ensemble-expert-proportion", po::value<float>()->default_value(0.f), "the proportion of expert policy ")
    ;

  return cmd;
}

EnsemblePostagDataGenerator::EnsemblePostagDataGenerator(std::vector<PostagModel*>& engines,
                                                         const po::variables_map & conf) : engines(engines) {
  _INFO << "[twpipe|postag|ensemble_generator] number of ensembled parsers: " << engines.size();

  n_samples = conf["ensemble-n-samples"].as<unsigned>();
  _INFO << "[twpipe|postag|ensemble_generator] generate " << n_samples << " for each instance.";

  std::string rollin_name = conf["ensemble-rollin"].as<std::string>();
  if (rollin_name == "expert") {
    rollin_policy = kExpert;
    proportion = conf["ensemble-expert-proportion"].as<float>();
    if (proportion > 1.) {
      proportion = 1.f;
      _INFO << "[twpipe|postag|ensemble_generator] proportion should be less than 1, reset.";
    } else if (proportion < 0.) {
      _INFO << "[twpipe|postag|ensemble_generator] proportion should be greater than 0., reset.";
    }
    _INFO << "[twpipe|parser|ensemble_generator] roll-in policy: " << rollin_name;
    _INFO << "[twpipe|parser|ensemble_generator] expert proportion: " << proportion;
  } else if (rollin_name == "predict") {
    rollin_policy = kPredict;
    _INFO << "[twpipe|parser|ensemble_generator] roll-in policy: " << rollin_name;
  } else {
    _ERROR << "[twpipe|parser|ensemble_generator] unknown roll-in policy: " << rollin_name;
    exit(1);
  }
}

void EnsemblePostagDataGenerator::generate(const std::vector<std::string> & words,
                                           const std::vector<std::string> & gold_postags,
                                           std::vector<unsigned>& pred_postags,
                                           std::vector<std::vector<float>>& prob) {
  Alphabet & pos_map = AlphabetCollection::get()->pos_map;
  unsigned n_engines = engines.size();
  dynet::ComputationGraph cg;

  for (unsigned i = 0; i < n_engines; ++i) {
    std::vector<std::vector<float>> single_probs;
    engines[i]->new_graph(cg);
    engines[i]->initialize(words);
  }

  unsigned n_words = words.size();
  unsigned prev_label = pos_map.get(Corpus::ROOT);

  prob.resize(n_words);
  for (unsigned i = 0; i < n_words; ++i) {
    std::vector<float> & ensembled_prob = prob[i];
    ensembled_prob.resize(pos_map.size(), 0.f);

    for (auto engine : engines) {
      dynet::Expression feature = engine->get_feature(i, prev_label);
      dynet::Expression logits = engine->get_emit_score(feature);
      std::vector<float> scores = dynet::as_vector(cg.get_value(logits));
      Math::softmax_inplace(scores);
      for (unsigned m = 0; m < pos_map.size(); ++m) { ensembled_prob[m] += scores[m]; }
    }
    for (unsigned m = 0; m < pos_map.size(); ++m) { ensembled_prob[m] /= n_engines; }
    
    unsigned label;
    if (rollin_policy == kExpert) {
      label = pos_map.get(gold_postags[i]);
    } else {
      label = std::max_element(ensembled_prob.begin(), ensembled_prob.end()) - ensembled_prob.begin();
    }
    pred_postags.push_back(label);
    prev_label = label;
  }
}

}
