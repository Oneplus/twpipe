#include "sampler.h"
#include "tree.h"
#include "twpipe/corpus.h"
#include "twpipe/math.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

OracleSampler::OracleSampler(ParseModel *engine) : engine(engine) {

}

void OracleSampler::sample(const std::vector<std::string> &words,
                           const std::vector<std::string> &postags,
                           const std::vector<unsigned> &heads,
                           const std::vector<std::string> &deprels,
                           std::vector<unsigned> &actions) {

  actions.clear();

  TransitionSystem &system = engine->sys;
  if (!DependencyUtils::is_tree(heads) ||
      (!system.allow_nonprojective() && DependencyUtils::is_non_projective(heads))) {
    return;
  }

  InputUnits input;
  Corpus::vector_to_input_units(words, postags, input);

  unsigned len = input.size();
  State state(len);

  engine->initialize_state(input, state);

  std::vector<unsigned> numeric_deprels(deprels.size());
  numeric_deprels[0] = Corpus::BAD_DEL;
  for (unsigned i = 1; i < deprels.size(); ++i) {
    numeric_deprels[i] = AlphabetCollection::get()->deprel_map.get(deprels[i]);
  }
  system.get_oracle_actions(heads, numeric_deprels, actions);
}

VanillaSampler::VanillaSampler(ParseModel *engine) : engine(engine) {

}

void VanillaSampler::sample(const std::vector<std::string> &words,
                            const std::vector<std::string> &postags,
                            const std::vector<unsigned> &heads,
                            const std::vector<std::string> &deprels,
                            std::vector<unsigned> &actions) {
  actions.clear();
  TransitionSystem & system = engine->sys;
  if (!DependencyUtils::is_tree(heads) ||
      (!system.allow_nonprojective() && DependencyUtils::is_non_projective(heads))) {
    return;
  }

  dynet::ComputationGraph cg;

  InputUnits input;
  Corpus::vector_to_input_units(words, postags, input);
  ParseModel::StateCheckpoint * checkpoint;

  unsigned len = input.size();
  State state(len);
  engine->initialize_state(input, state);

  engine->new_graph(cg);
  checkpoint = engine->get_initial_checkpoint();
  engine->initialize_parser(cg, input, checkpoint);

  unsigned n_actions = 0;
  while (!state.terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);

    dynet::Expression score_exprs = engine->get_scores(checkpoint);
    std::vector<float> probs = dynet::as_vector(cg.get_value(score_exprs));
    Math::softmax_inplace(probs);

    std::vector<float> valid_prob;
    for (unsigned act : valid_actions) {
      valid_prob.push_back(log(probs[act]));
    }
    // renormalize
    Math::softmax_inplace(valid_prob);
    unsigned index = Math::distribution_sample(valid_prob, (*dynet::rndeng));
    unsigned action = valid_actions[index];

    actions.push_back(action);
    system.perform_action(state, action);
    engine->perform_action(action, state, cg, checkpoint);

    n_actions++;
  }

  delete checkpoint;
}

EnsembleSampler::EnsembleSampler(std::vector<ParseModel *> &engines) : engines(engines) {

}

void EnsembleSampler::sample(const std::vector<std::string> &words,
                             const std::vector<std::string> &postags,
                             const std::vector<unsigned> &heads,
                             const std::vector<std::string> &deprels,
                             std::vector<unsigned> &actions) {
  actions.clear();
  TransitionSystem & system = engines[0]->sys;
  if (!DependencyUtils::is_tree(heads) ||
      (!system.allow_nonprojective() && DependencyUtils::is_non_projective(heads))) {
    return;
  }

  unsigned n_engines = engines.size();
  dynet::ComputationGraph cg;

  InputUnits input;
  Corpus::vector_to_input_units(words, postags, input);
  std::vector<ParseModel::StateCheckpoint *> checkpoints(n_engines, nullptr);

  unsigned len = input.size();
  State state(len);
  engines[0]->initialize_state(input, state);
  for (unsigned i = 0; i < n_engines; ++i) {
    engines[i]->new_graph(cg);
    checkpoints[i] = engines[i]->get_initial_checkpoint();
    engines[i]->initialize_parser(cg, input, checkpoints[i]);
  }

  unsigned n_actions = 0;
  while (!state.terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);

    std::vector<float> ensemble_probs(system.num_actions(), 0.f);
    for (unsigned i = 0; i < n_engines; ++i) {
      dynet::Expression score_exprs = engines[i]->get_scores(checkpoints[i]);
      std::vector<float> ensemble_score = dynet::as_vector(cg.get_value(score_exprs));
      Math::softmax_inplace(ensemble_score);
      for (unsigned c = 0; c < ensemble_score.size(); ++c) {
        ensemble_probs[c] += ensemble_score[c];
      }
    }
    for (auto & p : ensemble_probs) { p /= n_engines; }

    std::vector<float> valid_prob;
    for (unsigned act : valid_actions) {
      valid_prob.push_back(log(ensemble_probs[act]));
    }
    // renormalize
    Math::softmax_inplace(valid_prob);
    unsigned index = Math::distribution_sample(valid_prob, (*dynet::rndeng));
    unsigned action = valid_actions[index];

    actions.push_back(action);

    system.perform_action(state, action);
    for (unsigned i = 0; i < n_engines; ++i) {
      engines[i]->perform_action(action, state, cg, checkpoints[i]);
    }

    n_actions++;
  }

  for (unsigned i = 0; i < n_engines; ++i) { delete checkpoints[i]; }
}

}