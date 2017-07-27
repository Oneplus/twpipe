#include "parser_nohistory.h"
#include "dynet/expr.h"
#include "corpus.h"
#include "logging.h"
#include "arceager.h"
#include "arcstd.h"
#include "archybrid.h"
#include "swap.h"
#include <vector>
#include <random>

void ParserArchNoHistory::ArcEagerFunction::perform_action(const unsigned& action,
                                                           dynet::ComputationGraph& cg,
                                                           std::vector<dynet::expr::Expression>& stack,
                                                           std::vector<dynet::expr::Expression>& buffer,
                                                           dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                                                           dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                                                           Merge3Layer& composer,
                                                           dynet::expr::Expression& rel_expr) {
  if (ArcEager::is_shift(action)) {
    // SHITF: counting for the last GUARD
    BOOST_ASSERT_MSG(buffer.size() > 1,
                     "In parser.cc: When performing SHIFT, there should be one or more inputs in buffer.");
    const dynet::expr::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcEager::is_left(action)) {
    dynet::expr::Expression mod_expr, hed_expr;
    hed_expr = buffer.back();
    mod_expr = stack.back();

    stack.pop_back();
    buffer.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    q_pointer = q_lstm.get_head(q_pointer);
    buffer.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else if (ArcEager::is_right(action)) {
    dynet::expr::Expression mod_expr, hed_expr;
    mod_expr = buffer.back();
    hed_expr = stack.back();

    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    stack.push_back(mod_expr);
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else {
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
  }
}

void ParserArchNoHistory::ArcStandardFunction::perform_action(const unsigned& action,
                                                              dynet::ComputationGraph& cg,
                                                              std::vector<dynet::expr::Expression>& stack,
                                                              std::vector<dynet::expr::Expression>& buffer,
                                                              dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                                                              dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                                                              Merge3Layer& composer,
                                                              dynet::expr::Expression& rel_expr) {
  if (ArcStandard::is_shift(action)) {
    const dynet::expr::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcStandard::is_drop(action)) {
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else {
    dynet::expr::Expression mod_expr, hed_expr;
    if (ArcStandard::is_left(action)) {
      hed_expr = stack.back();
      mod_expr = stack[stack.size() - 2];
    } else {
      mod_expr = stack.back();
      hed_expr = stack[stack.size() - 2];
    }
    stack.pop_back(); stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);

    stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

void ParserArchNoHistory::ArcHybridFunction::perform_action(const unsigned& action,
                                                            dynet::ComputationGraph& cg,
                                                            std::vector<dynet::expr::Expression>& stack,
                                                            std::vector<dynet::expr::Expression>& buffer,
                                                            dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                                                            dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                                                            Merge3Layer& composer,
                                                            dynet::expr::Expression& rel_expr) {

  if (ArcHybrid::is_shift(action)) {
    const dynet::expr::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcStandard::is_drop(action)) {
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (ArcHybrid::is_left(action)) {
    dynet::expr::Expression mod_expr, hed_expr;
    hed_expr = buffer.back();
    mod_expr = stack.back();

    stack.pop_back();
    buffer.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    q_pointer = q_lstm.get_head(q_pointer);
    buffer.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else {
    dynet::expr::Expression mod_expr, hed_expr;
    hed_expr = stack[stack.size() - 2];
    mod_expr = stack.back();

    stack.pop_back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

void ParserArchNoHistory::SwapFunction::perform_action(const unsigned & action,
                                                       dynet::ComputationGraph & cg,
                                                       std::vector<dynet::expr::Expression>& stack,
                                                       std::vector<dynet::expr::Expression>& buffer,
                                                       dynet::LSTMBuilder & s_lstm, dynet::RNNPointer& s_pointer,
                                                       dynet::LSTMBuilder & q_lstm, dynet::RNNPointer& q_pointer,
                                                       Merge3Layer & composer,
                                                       dynet::expr::Expression & rel_expr) {
  if (Swap::is_shift(action)) {
    // SHITF: counting for the last GUARD
    BOOST_ASSERT_MSG(buffer.size() > 1,
                     "In parser.cc: When performing SHIFT, there should be one or more inputs in buffer.");
    const dynet::expr::Expression& buffer_front = buffer.back();
    stack.push_back(buffer_front);
    s_lstm.add_input(s_pointer, buffer_front);
    s_pointer = s_lstm.state();
    buffer.pop_back();
    q_pointer = q_lstm.get_head(q_pointer);
  } else if (Swap::is_swap(action)) {
    dynet::expr::Expression j_expr = stack.back();
    dynet::expr::Expression i_expr = stack[stack.size() - 2];

    stack.pop_back();
    stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);
    stack.push_back(j_expr);
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
    buffer.push_back(i_expr);
    q_lstm.add_input(q_pointer, buffer.back());
    q_pointer = q_lstm.state();
  } else {
    dynet::expr::Expression mod_expr, hed_expr;
    if (Swap::is_left(action)) {
      hed_expr = stack.back();
      mod_expr = stack[stack.size() - 2];
    } else {
      mod_expr = stack.back();
      hed_expr = stack[stack.size() - 2];
    }
    stack.pop_back(); stack.pop_back();
    s_pointer = s_lstm.get_head(s_pointer);
    s_pointer = s_lstm.get_head(s_pointer);

    stack.push_back(dynet::expr::tanh(composer.get_output(hed_expr, mod_expr, rel_expr)));
    s_lstm.add_input(s_pointer, stack.back());
    s_pointer = s_lstm.state();
  }
}

ParserArchNoHistory::ParserArchNoHistory(dynet::Model& m,
                                         unsigned size_w,
                                         unsigned dim_w,
                                         unsigned size_p,
                                         unsigned dim_p,
                                         unsigned size_t,
                                         unsigned dim_t,
                                         unsigned size_l,
                                         unsigned dim_l,
                                         unsigned n_layers,
                                         unsigned dim_lstm_in,
                                         unsigned dim_hidden,
                                         const std::string& system_name,
                                         TransitionSystem& system,
                                         const std::unordered_map<unsigned, std::vector<float>>& embedding) :
  Parser(m, system, system_name),
  s_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  q_lstm(n_layers, dim_lstm_in, dim_hidden, m),
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  preword_emb(m, size_t, dim_t, false),
  rel_emb(m, size_l, dim_l),
  merge_input(m, dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden),
  composer(m, dim_lstm_in, dim_lstm_in, dim_l, dim_lstm_in),
  scorer(m, dim_hidden, size_l),
  p_buffer_guard(m.add_parameters({ dim_lstm_in })),
  p_stack_guard(m.add_parameters({ dim_lstm_in })),
  pretrained(embedding),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  size_t(size_t), dim_t(dim_t),
  size_l(size_l), dim_l(dim_l),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {

  for (auto it : pretrained) {
    preword_emb.p_labels.initialize(it.first, it.second);
  }

  if (system_name == "arcstd") {
    sys_func = new ArcStandardFunction();
  } else if (system_name == "arceager") {
    sys_func = new ArcEagerFunction();
  } else if (system_name == "archybrid") {
    sys_func = new ArcHybridFunction();
  } else if (system_name == "swap") {
    sys_func = new SwapFunction();
  } else {
    _ERROR << "Main:: Unknown transition system: " << system_name;
    exit(1);
  }
}

void ParserArchNoHistory::perform_action(const unsigned& action,
                                         dynet::ComputationGraph& cg,
                                         State& state,
                                         Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  dynet::expr::Expression rel_repr = rel_emb.embed(action);
  sys_func->perform_action(action, cg, cp->stack, cp->buffer,
                           s_lstm, cp->s_pointer,
                           q_lstm, cp->q_pointer,
                           composer, rel_repr);
  sys.perform_action(state, action);
}

dynet::expr::Expression ParserArchNoHistory::get_scores(Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  return scorer.get_output(dynet::expr::rectify(merge.get_output(
    s_lstm.get_h(cp->s_pointer).back(),
    q_lstm.get_h(cp->q_pointer).back()
  )));
}

void ParserArchNoHistory::new_graph(dynet::ComputationGraph& cg) {
  s_lstm.new_graph(cg);
  q_lstm.new_graph(cg);

  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  preword_emb.new_graph(cg);
  rel_emb.new_graph(cg);
 
  merge_input.new_graph(cg);
  merge.new_graph(cg);
  composer.new_graph(cg);
  scorer.new_graph(cg); 

  buffer_guard = dynet::expr::parameter(cg, p_buffer_guard);
  stack_guard = dynet::expr::parameter(cg, p_stack_guard);
}

void ParserArchNoHistory::initialize_parser(dynet::ComputationGraph & cg,
                                            const InputUnits & input,
                                            Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  s_lstm.start_new_sequence();
  q_lstm.start_new_sequence();

  unsigned len = input.size();
  cp->stack.clear();
  cp->buffer.resize(len + 1);
  // Pay attention to this, if the guard word is handled here, there is no need
  // to insert it when loading the data.
  cp->buffer[0] = buffer_guard;
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;
    unsigned nid = input[i].nid;
    if (!pretrained.count(nid)) { nid = 0; }

    cp->buffer[len - i] = dynet::expr::rectify(merge_input.get_output(
      word_emb.embed(wid), pos_emb.embed(pid), preword_emb.embed(nid)
    ));
  }

  // push word into buffer in reverse order, pay attention to (i == len).
  for (unsigned i = 0; i <= len; ++i) {
    q_lstm.add_input(cp->buffer[i]);
  }
  
  s_lstm.add_input(stack_guard);
  cp->stack.push_back(stack_guard);
  cp->s_pointer = s_lstm.state();
  cp->q_pointer = q_lstm.state();
}

Parser::StateCheckpoint * ParserArchNoHistory::get_initial_checkpoint() {
  return new StateCheckpointImpl();
}

Parser::StateCheckpoint * ParserArchNoHistory::copy_checkpoint(StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  StateCheckpointImpl * new_checkpoint = new StateCheckpointImpl();
  new_checkpoint->s_pointer = cp->s_pointer;
  new_checkpoint->q_pointer = cp->q_pointer;
  new_checkpoint->stack = cp->stack;
  new_checkpoint->buffer = cp->buffer;
  return new_checkpoint;
}

void ParserArchNoHistory::destropy_checkpoint(StateCheckpoint * checkpoint) {
  delete dynamic_cast<StateCheckpointImpl *>(checkpoint);
}
