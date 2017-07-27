#include "parser_kiperwasser16.h"
#include "logging.h"

void ParserKiperwasser16::ArcEagerFunction::extract_feature(std::vector<dynet::expr::Expression>& encoded,
                                                            dynet::expr::Expression& empty,
                                                            ParserKiperwasser16::StateCheckpointImpl & cp,
                                                            const State& state) {
  // S1, S0, B0, B1
  // should do after sys.perform_action
  unsigned stack_size = state.stack.size();
  if (stack_size > 2) { cp.f0 = encoded[state.stack[stack_size - 2]]; } else { cp.f0 = empty; }
  if (stack_size > 1) { cp.f1 = encoded[state.stack[stack_size - 1]]; } else { cp.f1 = empty; }
  
  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f2 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f2 = empty; }
  if (buffer_size > 2) { cp.f3 = encoded[state.buffer[buffer_size - 2]]; } else { cp.f3 = empty; }
}

void ParserKiperwasser16::ArcStandardFunction::extract_feature(std::vector<dynet::expr::Expression>& encoded,
                                                               dynet::expr::Expression& empty,
                                                               ParserKiperwasser16::StateCheckpointImpl & cp,
                                                               const State& state) {
  // should considering the guard in state and buffer.
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }
  
  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; }
}

void ParserKiperwasser16::ArcHybridFunction::extract_feature(std::vector<dynet::expr::Expression>& encoded,
                                                             dynet::expr::Expression& empty,
                                                             ParserKiperwasser16::StateCheckpointImpl & cp,
                                                             const State& state) {
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; } }

void ParserKiperwasser16::SwapFunction::extract_feature(std::vector<dynet::expr::Expression>& encoded,
                                                        dynet::expr::Expression& empty,
                                                        ParserKiperwasser16::StateCheckpointImpl & cp,
                                                        const State& state) {
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; }
}

ParserKiperwasser16::ParserKiperwasser16(dynet::Model & m,
                                         unsigned size_w,
                                         unsigned dim_w,
                                         unsigned size_p,
                                         unsigned dim_p,
                                         unsigned size_t,
                                         unsigned dim_t,
                                         unsigned size_a,
                                         unsigned n_layers,
                                         unsigned dim_lstm_in,
                                         unsigned dim_hidden,
                                         const std::string & system_name,
                                         TransitionSystem & system,
                                         const std::unordered_map<unsigned, std::vector<float>>& embedding) : 
  Parser(m, system, system_name),
  fwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  bwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  preword_emb(m, size_t, dim_t, false),
  merge_input(m, dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden), 
  scorer(m, dim_hidden, size_a),
  p_empty(m.add_parameters({ dim_hidden })),
  p_fwd_guard(m.add_parameters({ dim_lstm_in })),
  p_bwd_guard(m.add_parameters({ dim_lstm_in })),
  sys_func(nullptr),
  pretrained(embedding),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  size_t(size_t), dim_t(dim_t),
  size_a(size_a),
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

void ParserKiperwasser16::new_graph(dynet::ComputationGraph & cg) {
  fwd_lstm.new_graph(cg);
  bwd_lstm.new_graph(cg);
  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  preword_emb.new_graph(cg);
  merge_input.new_graph(cg);
  merge.new_graph(cg);
  scorer.new_graph(cg);

  fwd_guard = dynet::expr::parameter(cg, p_fwd_guard);
  bwd_guard = dynet::expr::parameter(cg, p_bwd_guard);
  empty = dynet::expr::parameter(cg, p_empty);
}

void ParserKiperwasser16::initialize_parser(dynet::ComputationGraph & cg,
                                            const InputUnits & input,
                                            Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  fwd_lstm.start_new_sequence();
  bwd_lstm.start_new_sequence();

  unsigned len = input.size();
  std::vector<dynet::expr::Expression> lstm_input(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;
    unsigned aux_wid = input[i].aux_wid;
    if (!pretrained.count(aux_wid)) { aux_wid = 0; }

    lstm_input[i] = dynet::expr::rectify(merge_input.get_output(
      word_emb.embed(wid), pos_emb.embed(pid), preword_emb.embed(aux_wid)));
  }

  fwd_lstm.add_input(fwd_guard);
  bwd_lstm.add_input(bwd_guard);
  std::vector<dynet::expr::Expression> fwd_lstm_output(len);
  std::vector<dynet::expr::Expression> bwd_lstm_output(len);
  for (unsigned i = 0; i < len; ++i) {
    fwd_lstm.add_input(lstm_input[i]);
    bwd_lstm.add_input(lstm_input[len - 1 - i]);
    fwd_lstm_output[i] = fwd_lstm.back();
    bwd_lstm_output[len - 1 - i] = bwd_lstm.back();
  }
  encoded.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    encoded[i] = dynet::expr::concatenate({fwd_lstm_output[i], bwd_lstm_output[i]});
  }

  State state(len);
  initialize_state(input, state);
  sys_func->extract_feature(encoded, empty, *cp, state);
}

void ParserKiperwasser16::perform_action(const unsigned & action,
                                         dynet::ComputationGraph & cg,
                                         State & state,
                                         Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  sys.perform_action(state, action);
  sys_func->extract_feature(encoded, empty, *cp, state);
}

Parser::StateCheckpoint * ParserKiperwasser16::get_initial_checkpoint() {
  return new StateCheckpointImpl();
}

Parser::StateCheckpoint * ParserKiperwasser16::copy_checkpoint(StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  StateCheckpointImpl * new_checkpoint = new StateCheckpointImpl();
  new_checkpoint->f0 = cp->f0;
  new_checkpoint->f1 = cp->f1;
  new_checkpoint->f2 = cp->f2;
  new_checkpoint->f3 = cp->f3;
  return new_checkpoint;
}

void ParserKiperwasser16::destropy_checkpoint(StateCheckpoint * checkpoint) {
  delete dynamic_cast<StateCheckpointImpl *>(checkpoint);
}

dynet::expr::Expression ParserKiperwasser16::get_scores(Parser::StateCheckpoint * checkpoint) {
  StateCheckpointImpl * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  return scorer.get_output(dynet::expr::tanh(merge.get_output(cp->f0, cp->f1, cp->f2, cp->f3)));
}
