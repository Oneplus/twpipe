#include "parse_model.h"
#include "noisify.h"
#include "dynet/expr.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include <vector>
#include <random>

namespace twpipe {

std::pair<unsigned, float> ParseModel::get_best_action(const std::vector<float>& scores,
                                                       const std::vector<unsigned>& valid_actions) {
  unsigned best_a = valid_actions[0];
  float best_score = scores[best_a];
  //! should use next valid action.
  for (unsigned i = 1; i < valid_actions.size(); ++i) {
    unsigned a = valid_actions[i];
    if (best_score < scores[a]) {
      best_a = a;
      best_score = scores[a];
    }
  }
  return std::make_pair(best_a, best_score);
}

po::options_description ParseModel::get_options() {
  po::options_description cmd("Parser settings.");
  cmd.add_options()
    ("parse-arch", po::value<std::string>()->default_value("b15"), "The architecture [dyer15, ballesteros15, kiperwasser16].")
    ("parse-system", po::value<std::string>()->default_value("archybrid"), "")
    ("parse-n-layer", po::value<unsigned>()->default_value(2), "The number of layers in LSTM.")
    ("parse-char-dim", po::value<unsigned>()->default_value(16), "The dimension of char.")
    ("parse-word-dim", po::value<unsigned>()->default_value(32), "number of LSTM layers.")
    ("parse-pos-dim", po::value<unsigned>()->default_value(12), "POS dim, set it as 0 to disable POS.")
    ("parse-pretrained-dim", po::value<unsigned>()->default_value(100), "Pretrained input dimension.")
    ("parse-action-dim", po::value<unsigned>()->default_value(20), "The dimension for action.")
    ("parse-label-dim", po::value<unsigned>()->default_value(20), "The dimension for label.")
    ("parse-lstm-input-dim", po::value<unsigned>()->default_value(100), "The dimension for lstm input.")
    ("parse-hidden-dim", po::value<unsigned>()->default_value(100), "The dimension for hidden unit.")
    ;
  return cmd;
}

void ParseModel::initialize(dynet::ComputationGraph & cg,
                        const InputUnits & input,
                        State & state,
                        ParseModel::StateCheckpoint * checkpoint) {
  initialize_state(input, state);
  initialize_parser(cg, input, checkpoint);
}

void ParseModel::initialize_state(const InputUnits & input, State & state) {
  unsigned len = input.size();
  state.buffer.resize(len + 1);
  for (unsigned i = 0; i < len; ++i) { state.buffer[len - i] = i; }
  state.buffer[0] = Corpus::BAD_HED;
  state.stack.push_back(Corpus::BAD_HED);
}

ParseModel::ParseModel(dynet::ParameterCollection & m,
                       TransitionSystem & s) : model(m), sys(s) {
}

void ParseModel::predict(const std::vector<std::string>& words,
                         const std::vector<std::string>& postags,
                         std::vector<unsigned>& heads,
                         std::vector<std::string>& deprels) {
  InputUnits input;
  raw_to_input_units(words, postags, input);

  ParseUnits result;
  dynet::ComputationGraph cg;
  predict(cg, input, result);

  parse_units_to_raw(result, heads, deprels);
}

void ParseModel::label(const std::vector<std::string> & words,
                       const std::vector<std::string> & postags,
                       const std::vector<unsigned> & heads,
                       std::vector<std::string> & deprels) {
  BOOST_ASSERT_MSG(false, "not implemented.");
}

void ParseModel::raw_to_input_units(const std::vector<std::string>& words,
                                    const std::vector<std::string>& postags,
                                    InputUnits & units) {
  // The first element is the pseudo root.
  units.clear();

  Alphabet & word_map = AlphabetCollection::get()->word_map;
  Alphabet & char_map = AlphabetCollection::get()->char_map;
  Alphabet & pos_map = AlphabetCollection::get()->pos_map;

  InputUnit unit;
  unit.wid = word_map.get(Corpus::ROOT);
  unit.pid = pos_map.get(Corpus::ROOT);
  unit.aux_wid = unit.wid;
  unit.word = Corpus::ROOT;
  unit.postag = Corpus::ROOT;
  unit.lemma = Corpus::ROOT;
  unit.feature = Corpus::ROOT;
  units.push_back(unit);

  for (unsigned i = 0; i < words.size(); ++i) {
    const std::string & word = words[i];
    const std::string & postag = postags[i];

    unit.wid = (word_map.contains(word) ? word_map.get(word) : word_map.get(Corpus::UNK));
    unit.pid = pos_map.get(postag);
    unit.aux_wid = unit.wid;
    unit.word = word;
    unit.postag = postag;

    unsigned cur = 0;
    unit.cids.clear();
    while (cur < word.size()) {
      unsigned len = utf8_len(word[cur]);
      std::string ch_str = word.substr(cur, len);
      unit.cids.push_back(
        char_map.contains(ch_str) ? char_map.get(ch_str) : char_map.get(Corpus::UNK)
      );
      cur += len;
    }
    units.push_back(unit);
  }
}

void ParseModel::parse_units_to_raw(const ParseUnits & units,
                                    std::vector<unsigned>& heads,
                                    std::vector<std::string>& deprels,
                                    bool add_pseudo_root) {
  heads.clear();
  deprels.clear();

  if (add_pseudo_root) {
    heads.push_back(Corpus::BAD_HED);
    deprels.push_back(Corpus::ROOT);
  }
  // The first element in units is pseduo root.
  for (unsigned i = 1; i < units.size(); ++i) {
    heads.push_back(units[i].head);
    deprels.push_back(AlphabetCollection::get()->deprel_map.get(units[i].deprel));
  }
}

void ParseModel::predict(dynet::ComputationGraph& cg,
                         const InputUnits& input,
                         ParseUnits& parse) {
  new_graph(cg);

  unsigned len = input.size();
  State state(len);
  StateCheckpoint * checkpoint = get_initial_checkpoint();
  initialize(cg, input, state, checkpoint);

  std::vector<unsigned> actions;
  while (!state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    sys.get_valid_actions(state, valid_actions);

    dynet::Expression score_exprs = get_scores(checkpoint);
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    auto payload = get_best_action(scores, valid_actions);
    unsigned best_a = payload.first;
    actions.push_back(best_a);
    perform_action(best_a, cg, state, checkpoint);
  }
  destropy_checkpoint(checkpoint);
  vector_to_parse(state.heads, state.deprels, parse);
}

void ParseModel::label(dynet::ComputationGraph & cg,
                   const InputUnits & input,
                   const ParseUnits & parse,
                   ParseUnits & output) {
  new_graph(cg);

  unsigned len = input.size();
  State state(len);
  StateCheckpoint * checkpoint = get_initial_checkpoint();
  initialize(cg, input, state, checkpoint);

  std::vector<unsigned> ref_heads, ref_deprels;
  parse_to_vector(parse, ref_heads, ref_deprels);
  std::vector<unsigned> ref_actions;
  sys.get_oracle_actions(ref_heads, ref_deprels, ref_actions);
  unsigned step = 0;
  while (!state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    sys.get_valid_actions(state, valid_actions);

    dynet::Expression score_exprs = get_scores(checkpoint);
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));

    unsigned best_a = UINT_MAX, ref_structure_action = sys.get_structure_action(ref_actions[step]);
    for (unsigned i = 0; i < scores.size(); ++i) {
      if (ref_structure_action != sys.get_structure_action(i)) { continue; }
      if (best_a == UINT_MAX || scores[best_a] < scores[i]) { best_a = i; }
    }
    assert(best_a != UINT_MAX);
    perform_action(best_a, cg, state, checkpoint);
    step++;
  }
  destropy_checkpoint(checkpoint);
  vector_to_parse(state.heads, state.deprels, output);
}

void ParseModel::beam_search(dynet::ComputationGraph & cg,
                             const InputUnits & input,
                             const unsigned& beam_size,
                             bool structure_score,
                             std::vector<ParseUnits>& parses) {
  typedef std::tuple<unsigned, unsigned, float> Transition;

  new_graph(cg);
  unsigned len = input.size();
  std::vector<State> states;
  std::vector<float> scores;
  std::vector<StateCheckpoint*> checkpoints;

  states.push_back(State(len));
  scores.push_back(0.);
  StateCheckpoint * initial_checkpoint = get_initial_checkpoint();
  initialize(cg, input, states[0], initial_checkpoint);
  checkpoints.push_back(initial_checkpoint);

  unsigned curr = 0, next = 1;
  while (!states[curr].terminated()) {
    std::vector<Transition> transitions;
    for (unsigned i = curr; i < next; ++i) {
      const State& state = states[i];
      float score = scores[i];
      StateCheckpoint * checkpoint = checkpoints[i];

      std::vector<unsigned> valid_actions;
      sys.get_valid_actions(state, valid_actions);

      dynet::Expression score_exprs = get_scores(checkpoint);
      if (!structure_score) { score_exprs = dynet::log_softmax(score_exprs); }
      std::vector<float> s = dynet::as_vector(cg.get_value(score_exprs));
      for (unsigned a : valid_actions) {
        transitions.push_back(std::make_tuple(i, a, score + s[a]));
      }
    }

    sort(transitions.begin(), transitions.end(),
         [](const Transition& a, const Transition& b) { return std::get<2>(a) > std::get<2>(b); });
    curr = next;

    for (unsigned i = 0; i < transitions.size() && i < beam_size; ++i) {
      unsigned cursor = std::get<0>(transitions[i]);
      unsigned action = std::get<1>(transitions[i]);
      float new_score = std::get<2>(transitions[i]);


      State& state = states[cursor];
      State new_state(state);
      StateCheckpoint * new_checkpoint = copy_checkpoint(checkpoints[cursor]);
      perform_action(action, cg, new_state, new_checkpoint);

      //      
      states.push_back(new_state);
      scores.push_back(new_score);
      checkpoints.push_back(new_checkpoint);
      next++;
    }
  }
  for (StateCheckpoint * checkpoint : checkpoints) {
    destropy_checkpoint(checkpoint);
  }
  parses.resize(next - curr);
  for (unsigned i = curr; i < next; ++i) {
    vector_to_parse(states[i].heads, states[i].deprels, parses[i - curr]);
  }
}

}