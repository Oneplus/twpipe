#include "tester.h"
#include "twpipe/corpus.h"
#include "twpipe/math.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

OracleTester::OracleTester(ParseModel *engine) : engine(engine) {

}

void OracleTester::test(const std::vector<std::string> &words,
                        const std::vector<std::string> &postags,
                        const std::vector<unsigned> &heads,
                        const std::vector<std::string> &deprels,
                        const std::vector<unsigned> &actions,
                        std::vector<std::vector<float>> &probs) {

  InputUnits input;
  Corpus::vector_to_input_units(words, postags, input);

  unsigned len = input.size();
  State state(len);

  engine->initialize_state(input, state);
  probs.clear();

  TransitionSystem &system = engine->sys;

  unsigned n_actions = 0;

  std::vector<unsigned> numeric_deprels(deprels.size());
  numeric_deprels[0] = Corpus::BAD_DEL;
  for (unsigned i = 1; i < deprels.size(); ++i) {
    numeric_deprels[i] = AlphabetCollection::get()->deprel_map.get(deprels[i]);
  }

  while (!state.terminated()) {
    std::vector<unsigned> valid_actions;
    system.get_valid_actions(state, valid_actions);

    std::vector<float> costs;
    std::vector<float> prob(system.num_actions(), 0.f);
    system.get_transition_costs(state, valid_actions, heads, numeric_deprels, costs);

    unsigned best_cost_offset = std::max_element(costs.begin(), costs.end()) - costs.begin();
    float best_costs = costs[best_cost_offset];

    for (unsigned j = 0; j < valid_actions.size(); ++j) {
      if (costs[j] == best_costs) {
        prob[valid_actions[j]] = 1.f;
      }
    }

    unsigned action = actions[n_actions];
    probs.push_back(prob);
    system.perform_action(state, action);
    ++n_actions;
  }
}

VanillaTester::VanillaTester(ParseModel *engine) : engine(engine) {

}

void VanillaTester::test(const std::vector<std::string> &words,
                         const std::vector<std::string> &postags,
                         const std::vector<unsigned> &heads,
                         const std::vector<std::string> &deprels,
                         const std::vector<unsigned> &actions,
                         std::vector<std::vector<float>> &probs) {
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

  probs.clear();
  TransitionSystem & system = engine->sys;

  unsigned n_actions = 0;
  while (!state.terminated()) {
    dynet::Expression score_exprs = engine->get_scores(checkpoint);
    std::vector<float> prob = dynet::as_vector(cg.get_value(score_exprs));
    Math::softmax_inplace(prob);

    unsigned action = actions[n_actions];

    probs.push_back(prob);
    system.perform_action(state, action);
    engine->perform_action(action, state, cg, checkpoint);

    n_actions++;
  }

  delete checkpoint;
}

EnsembleTester::EnsembleTester(std::vector<ParseModel *> &engines) : engines(engines) {

}

void EnsembleTester::test(const std::vector<std::string> &words,
                          const std::vector<std::string> &postags,
                          const std::vector<unsigned> &heads,
                          const std::vector<std::string> &deprels,
                          const std::vector<unsigned> &actions,
                          std::vector<std::vector<float>> &probs) {
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

  probs.clear();
  TransitionSystem & system = engines[0]->sys;

  unsigned n_actions = 0;
  while (!state.terminated()) {
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

    unsigned action = actions[n_actions];

    probs.push_back(ensemble_probs);
    system.perform_action(state, action);
    for (unsigned i = 0; i < n_engines; ++i) {
      engines[i]->perform_action(action, state, cg, checkpoints[i]);
    }

    n_actions++;
  }

  for (unsigned i = 0; i < n_engines; ++i) { delete checkpoints[i]; }
}

}