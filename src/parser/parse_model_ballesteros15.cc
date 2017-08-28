#include "parse_model_ballesteros15.h"
#include "dynet/expr.h"
#include "archybrid.h"
// #include "arceager.h"
// #include "arcstd.h"
// #include "swap.h"
#include "twpipe/corpus.h"
#include "twpipe/logging.h"
#include "twpipe/embedding.h"
#include "twpipe/alphabet_collection.h"
#include <vector>
#include <random>

namespace twpipe {
/*
void ParserBallesteros15::ArcEagerFunction::perform_action(const unsigned& action,
                                                           dynet::ComputationGraph& cg,
                                                           dynet::LSTMBuilder& a_lstm,
                                                           dynet::LSTMBuilder& s_lstm,
                                                           dynet::LSTMBuilder& q_lstm,
                                                           Merge3Layer& composer,
                                                           ParserBallesteros15::StateCheckpointImpl & cp,
                                                           dynet::Expression& act_expr,
                                                           dynet::Expression& rel_expr) {
  a_lstm.add_input(cp.a_pointer, act_expr);
  cp.a_pointer = a_lstm.state();

  if (ArcEager::is_shift(action)) {
    const dynet::Expression& buffer_front = cp.buffer.back();
    cp.stack.push_back(buffer_front);
    s_lstm.add_input(cp.s_pointer, buffer_front);
    cp.s_pointer = s_lstm.state();
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else if (ArcEager::is_left(action)) {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = cp.buffer.back();
    mod_expr = cp.stack.back();

    cp.stack.pop_back();
    cp.buffer.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
    cp.buffer.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(cp.q_pointer, cp.buffer.back());
    cp.q_pointer = q_lstm.state();
  } else if (ArcEager::is_right(action)) {
    dynet::Expression mod_expr, hed_expr;
    mod_expr = cp.buffer.back();
    hed_expr = cp.stack.back();

    cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
    cp.stack.push_back(mod_expr);
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else {
    cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
  }
}
*/
/*
void ParserBallesteros15::ArcStandardFunction::perform_action(const unsigned& action,
                                                              dynet::ComputationGraph& cg,
                                                              dynet::LSTMBuilder& a_lstm,
                                                              dynet::LSTMBuilder& s_lstm,
                                                              dynet::LSTMBuilder& q_lstm,
                                                              Merge3Layer& composer,
                                                              ParserBallesteros15::StateCheckpointImpl & cp,
                                                              dynet::Expression& act_expr,
                                                              dynet::Expression& rel_expr) {
  a_lstm.add_input(cp.a_pointer, act_expr);
  cp.a_pointer = a_lstm.state();
  if (ArcStandard::is_shift(action)) {
    const dynet::expr::Expression& buffer_front = cp.buffer.back();
    cp.stack.push_back(buffer_front);
    s_lstm.add_input(cp.s_pointer, buffer_front);
    cp.s_pointer = s_lstm.state();
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else if (ArcStandard::is_drop(action)) {
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else {
    dynet::Expression mod_expr, hed_expr;
    if (ArcStandard::is_left(action)) {
      hed_expr = cp.stack.back();
      mod_expr = cp.stack[cp.stack.size() - 2];
    } else {
      mod_expr = cp.stack.back();
      hed_expr = cp.stack[cp.stack.size() - 2];
    }
    cp.stack.pop_back(); cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);

    cp.stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
  }
}
*/

void Ballesteros15Model::ArcHybridFunction::perform_action(const unsigned& action,
                                                           dynet::ComputationGraph& cg,
                                                           Ballesteros15Model::LSTMBuilderType & a_lstm,
                                                           Ballesteros15Model::LSTMBuilderType & s_lstm,
                                                           Ballesteros15Model::LSTMBuilderType & q_lstm,
                                                           Merge3Layer& composer,
                                                           Ballesteros15Model::StateCheckpointImpl & cp,
                                                           dynet::Expression& act_expr,
                                                           dynet::Expression& rel_expr) {
  a_lstm.add_input(cp.a_pointer, act_expr);
  cp.a_pointer = a_lstm.state();
  if (ArcHybrid::is_shift(action)) {
    const dynet::Expression& buffer_front = cp.buffer.back();
    cp.stack.push_back(buffer_front);
    s_lstm.add_input(cp.s_pointer, buffer_front);
    cp.s_pointer = s_lstm.state();
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else if (ArcHybrid::is_left(action)) {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = cp.buffer.back();
    mod_expr = cp.stack.back();

    cp.stack.pop_back();
    cp.buffer.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
    cp.buffer.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(cp.q_pointer, cp.buffer.back());
    cp.q_pointer = q_lstm.state();
  } else {
    dynet::Expression mod_expr, hed_expr;
    hed_expr = cp.stack[cp.stack.size() - 2];
    mod_expr = cp.stack.back();

    cp.stack.pop_back();
    cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.stack.push_back(dynet::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
  }
}

/*
void ParserBallesteros15::SwapFunction::perform_action(const unsigned & action,
                                                       dynet::ComputationGraph & cg,
                                                       dynet::LSTMBuilder & a_lstm,
                                                       dynet::LSTMBuilder & s_lstm,
                                                       dynet::LSTMBuilder & q_lstm,
                                                       Merge3Layer & composer,
                                                       ParserBallesteros15::StateCheckpointImpl & cp,
                                                       dynet::expr::Expression & act_expr,
                                                       dynet::expr::Expression & rel_expr) {
  a_lstm.add_input(cp.a_pointer, act_expr);
  cp.a_pointer = a_lstm.state();
  if (Swap::is_shift(action)) {
    const dynet::expr::Expression& buffer_front = cp.buffer.back();
    cp.stack.push_back(buffer_front);
    s_lstm.add_input(cp.s_pointer, buffer_front);
    cp.s_pointer = s_lstm.state();
    cp.buffer.pop_back();
    cp.q_pointer = q_lstm.get_head(cp.q_pointer);
  } else if (Swap::is_swap(action)) {
    dynet::expr::Expression j_expr = cp.stack.back();
    dynet::expr::Expression i_expr = cp.stack[cp.stack.size() - 2];

    cp.stack.pop_back();
    cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.stack.push_back(j_expr);
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
    cp.buffer.push_back(i_expr);
    q_lstm.add_input(cp.q_pointer, cp.buffer.back());
    cp.q_pointer = q_lstm.state();
  } else {
    dynet::expr::Expression mod_expr, hed_expr;
    if (Swap::is_left(action)) {
      hed_expr = cp.stack.back();
      mod_expr = cp.stack[cp.stack.size() - 2];
    } else {
      hed_expr = cp.stack[cp.stack.size() - 2];
      mod_expr = cp.stack.back();
    }
    cp.stack.pop_back();
    cp.stack.pop_back();
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.s_pointer = s_lstm.get_head(cp.s_pointer);
    cp.stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(cp.s_pointer, cp.stack.back());
    cp.s_pointer = s_lstm.state();
  }
}
*/
Ballesteros15Model::Ballesteros15Model(dynet::ParameterCollection & m,
                                       unsigned size_c,
                                       unsigned dim_c,
                                       unsigned dim_w,
                                       unsigned size_p,
                                       unsigned dim_p,
                                       unsigned dim_t,
                                       unsigned size_a,
                                       unsigned dim_a,
                                       unsigned dim_l,
                                       unsigned n_layers,
                                       unsigned dim_lstm_in,
                                       unsigned dim_hidden,
                                       TransitionSystem& system) :
  ParseModel(m, system),
  fwd_ch_lstm(1, dim_c, dim_w, m),  /* We mannually fix the n-layers of char lstm to 1. */
  bwd_ch_lstm(1, dim_c, dim_w, m),
  s_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  q_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  a_lstm(n_layers, dim_a, dim_hidden, m),
  char_emb(m, size_c, dim_c),
  pos_emb(m, size_p, dim_p),
  act_emb(m, size_a, dim_a),
  rel_emb(m, size_a, dim_l),
  pretrain_emb(dim_t),
  merge_input(m, dim_w + dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
  composer(m, dim_lstm_in, dim_lstm_in, dim_l, dim_lstm_in),
  scorer(m, dim_hidden, size_a),
  p_action_start(m.add_parameters({ dim_a })),
  p_buffer_guard(m.add_parameters({ dim_lstm_in })),
  p_stack_guard(m.add_parameters({ dim_lstm_in })),
  p_word_start_guard(m.add_parameters({ dim_c })),
  p_word_end_guard(m.add_parameters({ dim_c })),
  p_root_word(m.add_parameters({ dim_w + dim_w })),
  sys_func(nullptr),
  size_c(size_c), dim_c(dim_c), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  dim_t(dim_t),
  size_a(size_a), dim_a(dim_a), dim_l(dim_l),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {

  std::string system_name = system.name();

  if (system_name == "arcstd") {
    // sys_func = new ArcStandardFunction();
  } else if (system_name == "arceager") {
    // sys_func = new ArcEagerFunction();
  } else if (system_name == "archybrid") {
    sys_func = new ArcHybridFunction();
  } else if (system_name == "swap") {
    // sys_func = new SwapFunction();
  } else {
    _ERROR << "Main:: Unknown transition system: " << system_name;
    exit(1);
  }
}

void Ballesteros15Model::perform_action(const unsigned& action,
                                         dynet::ComputationGraph& cg,
                                         State& state,
                                         ParseModel::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  dynet::Expression act_repr = act_emb.embed(action);
  dynet::Expression rel_repr = rel_emb.embed(action);
  sys_func->perform_action(action, cg, a_lstm, s_lstm, q_lstm, composer, *cp, act_repr, rel_repr);
  sys.perform_action(state, action);
}

dynet::Expression Ballesteros15Model::get_scores(ParseModel::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  return scorer.get_output(dynet::rectify(merge.get_output(
    s_lstm.get_h(cp->s_pointer).back(),
    q_lstm.get_h(cp->q_pointer).back(),
    a_lstm.get_h(cp->a_pointer).back())
  ));
}

void Ballesteros15Model::raw_to_input_units(const std::vector<std::string>& words,
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

ParseModel::StateCheckpoint * Ballesteros15Model::get_initial_checkpoint() {
  return new StateCheckpointImpl();
}

ParseModel::StateCheckpoint * Ballesteros15Model::copy_checkpoint(ParseModel::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  StateCheckpointImpl * new_checkpoint = new StateCheckpointImpl();
  new_checkpoint->s_pointer = cp->s_pointer;
  new_checkpoint->q_pointer = cp->q_pointer;
  new_checkpoint->a_pointer = cp->a_pointer;
  new_checkpoint->stack = cp->stack;
  new_checkpoint->buffer = cp->buffer;
  return new_checkpoint;
}

void Ballesteros15Model::destropy_checkpoint(StateCheckpoint * checkpoint) {
  delete dynamic_cast<StateCheckpointImpl *>(checkpoint);
}

void Ballesteros15Model::new_graph(dynet::ComputationGraph& cg) {
  fwd_ch_lstm.new_graph(cg);
  bwd_ch_lstm.new_graph(cg);
  s_lstm.new_graph(cg);
  q_lstm.new_graph(cg);
  a_lstm.new_graph(cg);

  char_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  act_emb.new_graph(cg);
  rel_emb.new_graph(cg);
  pretrain_emb.new_graph(cg);

  merge_input.new_graph(cg);
  merge.new_graph(cg);
  composer.new_graph(cg);
  scorer.new_graph(cg);

  action_start = dynet::parameter(cg, p_action_start);
  buffer_guard = dynet::parameter(cg, p_buffer_guard);
  stack_guard = dynet::parameter(cg, p_stack_guard);
  word_start_guard = dynet::parameter(cg, p_word_start_guard);
  word_end_guard = dynet::parameter(cg, p_word_end_guard);
  root_word = dynet::parameter(cg, p_root_word);
}

void Ballesteros15Model::initialize_parser(dynet::ComputationGraph & cg,
                                           const InputUnits & input,
                                           ParseModel::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);

  std::vector<std::vector<float>> embeddings;
  unsigned len = input.size();
  std::vector<std::string> words(len);
  // The first unit is pseduo root.
  for (unsigned i = 0; i < len; ++i) { words[i] = input[i].word; }
  WordEmbedding::get()->render(words, embeddings);

  s_lstm.start_new_sequence();
  q_lstm.start_new_sequence();
  a_lstm.start_new_sequence();
  a_lstm.add_input(action_start);

  cp->stack.clear();
  cp->buffer.resize(len + 1);

  // Pay attention to this, if the guard word is handled here, there is no need
  // to insert it when loading the data.
  cp->buffer[0] = buffer_guard;
  for (unsigned i = 0; i < len; ++i) {
    unsigned pid = input[i].pid;

    dynet::Expression word_expr;
    if (i == 0) {
      /// first element, the root.
      word_expr = root_word;
    } else {
      fwd_ch_lstm.start_new_sequence();
      bwd_ch_lstm.start_new_sequence();
      fwd_ch_lstm.add_input(word_start_guard);
      bwd_ch_lstm.add_input(word_end_guard);
      unsigned n_chars = input[i].cids.size();
      for (unsigned j = 0; j < n_chars; ++j) {
        fwd_ch_lstm.add_input(char_emb.embed(input[i].cids[j]));
        bwd_ch_lstm.add_input(char_emb.embed(input[i].cids[n_chars - j - 1]));
      }
      fwd_ch_lstm.add_input(word_end_guard);
      bwd_ch_lstm.add_input(word_start_guard);
      word_expr = dynet::concatenate({ fwd_ch_lstm.back(), bwd_ch_lstm.back() });
    }
    cp->buffer[len - i] = dynet::rectify(merge_input.get_output(
      word_expr, pos_emb.embed(pid), pretrain_emb.get_output(embeddings[i])
    ));
  }

  // push word into buffer in reverse order, pay attention to (i == len).
  for (unsigned i = 0; i <= len; ++i) {
    q_lstm.add_input(cp->buffer[i]);
  }

  s_lstm.add_input(stack_guard);
  stack.push_back(stack_guard);
  cp->a_pointer = a_lstm.state();
  cp->s_pointer = s_lstm.state();
  cp->q_pointer = q_lstm.state();
}

}