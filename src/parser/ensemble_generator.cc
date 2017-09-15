#include "ensemble_generator.h"
#include "tree.h"
#include "twpipe/logging.h"
#include "twpipe/math.h"
#include "twpipe/corpus.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

po::options_description EnsembleParseDataGenerator::get_options() {
  po::options_description cmd("Ensemble data generate options.");
  cmd.add_options()
    ("ensemble-method", po::value<std::string>()->default_value("prob"), "ensemble methods [prob|logits_mean|logits_sum]")
    ("ensemble-n-samples", po::value<unsigned>()->default_value(1), "the number of samples.")
    ("ensemble-rollin", po::value<std::string>()->default_value("boltzmann"), "the rollin-method [expert|egreedy|boltzmann]")
    ("ensemble-expert-proportion", po::value<float>()->default_value(0.f), "the proportion of expert policy ")
    ("ensemble-egreedy-epsilon", po::value<float>()->default_value(0.1f), "the epsilon of epsilon-greedy policy.")
    ("ensemble-boltzmann-temperature", po::value<float>()->default_value(1.f), "the temperature of epsilon-greedy policy.")
    ;

  return cmd;
}

EnsembleParseDataGenerator::EnsembleParseDataGenerator(std::vector<ParseModel*>& engines,
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
  if (rollin_name == "expert") {
    rollin_policy = kExpert;
    proportion = conf["ensemble-expert-proportion"].as<float>();
    if (proportion > 1.) {
      proportion = 1.;
      _INFO << "[twpipe|parser|ensemble_generator] proportion should be less than 1, reset.";
    } else if (proportion < 0.) {
      _INFO << "[twpipe|parser|ensemble_generator] proportion should be greater than 0., reset.";
    }
    _INFO << "[twpipe|parser|ensemble_generator] roll-in policy: " << rollin_name;
    _INFO << "[twpipe|parser|ensemble_generator] expert proportion: " << proportion;
  } else if (rollin_name == "egreedy") {
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

void EnsembleParseDataGenerator::generate(const std::vector<std::string>& words,
                                          const std::vector<std::string>& postags,
                                          const std::vector<unsigned> & heads,
                                          const std::vector<std::string> & deprels,
                                          std::vector<unsigned>& actions,
                                          std::vector<std::vector<float>>& prob) {
  unsigned n_engines = engines.size();
  dynet::ComputationGraph cg;

  InputUnits input;
  ParseModel::raw_to_input_units(words, postags, input);
  std::vector<ParseModel::StateCheckpoint *> checkpoints(n_engines, nullptr);

  unsigned len = input.size();
  State state(len);
  engines[0]->initialize_state(input, state);
  for (unsigned i = 0; i < n_engines; ++i) {
    engines[i]->new_graph(cg);
    checkpoints[i] = engines[i]->get_initial_checkpoint();
    engines[i]->initialize_parser(cg, input, checkpoints[i]);
  }

  actions.clear();
  prob.clear();
  TransitionSystem & system = engines[0]->sys;

  std::vector<unsigned> gold_actions;
  if (rollin_policy == kExpert) {
    if (!DependencyUtils::is_tree(heads) ||
      (!system.allow_nonprojective() && DependencyUtils::is_non_projective(heads))) {
      for (unsigned i = 0; i < n_engines; ++i) { delete checkpoints[i]; }
      return;
    }
    std::vector<unsigned> numeric_deprels(deprels.size());
    numeric_deprels[0] = Corpus::BAD_DEL;
    for (unsigned i = 1; i < deprels.size(); ++i) {
      numeric_deprels[i] = AlphabetCollection::get()->deprel_map.get(deprels[i]);
    }
    system.get_oracle_actions(heads, numeric_deprels, gold_actions);
  }

  unsigned n_actions = 0;
  while (!state.terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);
   
    std::vector<float> ensemble_probs(system.num_actions(), 0.f);
    for (unsigned i = 0; i < n_engines; ++i) {
      dynet::Expression score_exprs = engines[i]->get_scores(checkpoints[i]);
      std::vector<float> ensemble_score = dynet::as_vector(cg.get_value(score_exprs));
      if (ensemble_method == kProbability) {
        Math::softmax_inplace(ensemble_score);
      }
      for (unsigned c = 0; c < ensemble_score.size(); ++c) {
        ensemble_probs[c] += ensemble_score[c];
      }
    }

    if (ensemble_method == kProbability || ensemble_method == kLogitsMean) {
      for (auto & p : ensemble_probs) { p /= n_engines; }
    }
    if (ensemble_method == kLogitsMean || ensemble_method == kLogitsSum) {
      Math::softmax_inplace(ensemble_probs);
    }

    unsigned action = UINT_MAX;
    if (rollin_policy == kExpert) {
      action = gold_actions[n_actions];
      for (unsigned i = 0; i < ensemble_probs.size(); ++i) {
        ensemble_probs[i] *= (1 - proportion);
        if (i == action) { ensemble_probs[i] += proportion; }
      }
    } else if (rollin_policy == kEpsilonGreedy) {
      float seed = dynet::rand01();
      if (seed < epsilon) {
        action = valid_actions[dynet::rand0n(valid_actions.size())];
      } else {
        auto payload = ParseModel::get_best_action(ensemble_probs, valid_actions);
        action = payload.first;
      }
    } else {
      std::vector<float> valid_prob;
      for (unsigned act : valid_actions) {
        valid_prob.push_back(log(ensemble_probs[act]) / temperature);
      }
      // renormalize
      Math::softmax_inplace(valid_prob);
      unsigned index = Math::distribution_sample(valid_prob, (*dynet::rndeng));
      action = valid_actions[index];
    }

    actions.push_back(action);
    prob.push_back(ensemble_probs);

    system.perform_action(state, action);
    for (unsigned i = 0; i < n_engines; ++i) {
      engines[i]->perform_action(action, state, cg, checkpoints[i]);
    }

    n_actions++;
  }

  for (unsigned i = 0; i < n_engines; ++i) { delete checkpoints[i]; }
}

}
