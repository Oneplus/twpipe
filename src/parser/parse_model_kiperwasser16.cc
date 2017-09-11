#include "parse_model_kiperwasser16.h"
#include "twpipe/logging.h"
#include "twpipe/embedding.h"

namespace twpipe {

void Kiperwasser16Model::ArcEagerFunction::extract_feature(std::vector<dynet::Expression>& encoded,
                                                           dynet::Expression& empty,
                                                           Kiperwasser16Model::StateCheckpointImpl & cp,
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

void Kiperwasser16Model::ArcStandardFunction::extract_feature(std::vector<dynet::Expression> & encoded,
                                                              dynet::Expression & empty,
                                                              Kiperwasser16Model::StateCheckpointImpl & cp,
                                                              const State& state) {
  // should considering the guard in state and buffer.
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; }
}

void Kiperwasser16Model::ArcHybridFunction::extract_feature(std::vector<dynet::Expression>& encoded,
                                                            dynet::Expression & empty,
                                                            Kiperwasser16Model::StateCheckpointImpl & cp,
                                                            const State& state) {
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; }
}

void Kiperwasser16Model::SwapFunction::extract_feature(std::vector<dynet::Expression>& encoded,
                                                       dynet::Expression& empty,
                                                       Kiperwasser16Model::StateCheckpointImpl & cp,
                                                       const State& state) {
  unsigned stack_size = state.stack.size();
  if (stack_size > 3) { cp.f0 = encoded[state.stack[stack_size - 3]]; } else { cp.f0 = empty; }
  if (stack_size > 2) { cp.f1 = encoded[state.stack[stack_size - 2]]; } else { cp.f1 = empty; }
  if (stack_size > 1) { cp.f2 = encoded[state.stack[stack_size - 1]]; } else { cp.f2 = empty; }

  unsigned buffer_size = state.buffer.size();
  if (buffer_size > 1) { cp.f3 = encoded[state.buffer[buffer_size - 1]]; } else { cp.f3 = empty; }
}

Kiperwasser16Model::Kiperwasser16Model(dynet::ParameterCollection & m,
                                       unsigned size_w,
                                       unsigned dim_w,
                                       unsigned size_p,
                                       unsigned dim_p,
                                       unsigned dim_t,
                                       unsigned size_a,
                                       unsigned n_layers,
                                       unsigned dim_lstm_in,
                                       unsigned dim_hidden,
                                       TransitionSystem & system) :
  ParseModel(m, system),
  fwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  bwd_lstm(n_layers, dim_lstm_in, dim_hidden / 2, m),
  word_emb(m, size_w, dim_w),
  pos_emb(m, size_p, dim_p),
  pretrain_emb(dim_t),
  merge_input(m, dim_w, dim_p, dim_t, dim_lstm_in),
  merge(m, dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
  scorer(m, dim_hidden, size_a),
  p_empty(m.add_parameters({ dim_hidden })),
  p_fwd_guard(m.add_parameters({ dim_lstm_in })),
  p_bwd_guard(m.add_parameters({ dim_lstm_in })),
  sys_func(nullptr),
  size_w(size_w), dim_w(dim_w),
  size_p(size_p), dim_p(dim_p),
  dim_t(dim_t),
  size_a(size_a),
  n_layers(n_layers), dim_lstm_in(dim_lstm_in), dim_hidden(dim_hidden) {

  std::string system_name = system.name();
  if (system_name == "arcstd") {
    sys_func = new ArcStandardFunction();
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

void Kiperwasser16Model::new_graph(dynet::ComputationGraph & cg) {
  fwd_lstm.new_graph(cg);
  bwd_lstm.new_graph(cg);
  word_emb.new_graph(cg);
  pos_emb.new_graph(cg);
  pretrain_emb.new_graph(cg);
  merge_input.new_graph(cg);
  merge.new_graph(cg);
  scorer.new_graph(cg);

  fwd_guard = dynet::parameter(cg, p_fwd_guard);
  bwd_guard = dynet::parameter(cg, p_bwd_guard);
  empty = dynet::parameter(cg, p_empty);
}

void Kiperwasser16Model::initialize_parser(dynet::ComputationGraph & cg,
                                           const InputUnits & input,
                                           ParseModel::StateCheckpoint * checkpoint) {
  auto * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);

  std::vector<std::vector<float>> embeddings;
  unsigned len = input.size();
  std::vector<std::string> words(len);
  // The first unit is pseduo root.
  for (unsigned i = 0; i < len; ++i) { words[i] = input[i].word; }
  WordEmbedding::get()->render(words, embeddings);

  fwd_lstm.start_new_sequence();
  bwd_lstm.start_new_sequence();

  std::vector<dynet::Expression> lstm_input(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = input[i].wid;
    unsigned pid = input[i].pid;

    lstm_input[i] = dynet::rectify(merge_input.get_output(
      word_emb.embed(wid), pos_emb.embed(pid), pretrain_emb.get_output(embeddings[i])));
  }

  fwd_lstm.add_input(fwd_guard);
  bwd_lstm.add_input(bwd_guard);
  std::vector<dynet::Expression> fwd_lstm_output(len);
  std::vector<dynet::Expression> bwd_lstm_output(len);
  for (unsigned i = 0; i < len; ++i) {
    fwd_lstm.add_input(lstm_input[i]);
    bwd_lstm.add_input(lstm_input[len - 1 - i]);
    fwd_lstm_output[i] = fwd_lstm.back();
    bwd_lstm_output[len - 1 - i] = bwd_lstm.back();
  }
  encoded.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    encoded[i] = dynet::concatenate({ fwd_lstm_output[i], bwd_lstm_output[i] });
  }

  State state(len);
  initialize_state(input, state);
  sys_func->extract_feature(encoded, empty, *cp, state);
}

void Kiperwasser16Model::perform_action(const unsigned & action,
                                        const State & state,
                                        dynet::ComputationGraph & cg,
                                        ParseModel::StateCheckpoint * checkpoint) {
  auto * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  sys_func->extract_feature(encoded, empty, *cp, state);
}

ParseModel::StateCheckpoint * Kiperwasser16Model::get_initial_checkpoint() {
  return new StateCheckpointImpl();
}

ParseModel::StateCheckpoint * Kiperwasser16Model::copy_checkpoint(StateCheckpoint * checkpoint) {
  auto * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  auto * new_checkpoint = new StateCheckpointImpl();
  new_checkpoint->f0 = cp->f0;
  new_checkpoint->f1 = cp->f1;
  new_checkpoint->f2 = cp->f2;
  new_checkpoint->f3 = cp->f3;
  return new_checkpoint;
}

void Kiperwasser16Model::destropy_checkpoint(StateCheckpoint * checkpoint) {
  delete dynamic_cast<StateCheckpointImpl *>(checkpoint);
}

dynet::Expression Kiperwasser16Model::get_scores(ParseModel::StateCheckpoint * checkpoint) {
  auto * cp = dynamic_cast<StateCheckpointImpl *>(checkpoint);
  return scorer.get_output(dynet::tanh(merge.get_output(cp->f0, cp->f1, cp->f2, cp->f3)));
}

dynet::Expression Kiperwasser16Model::l2() {
  std::vector<dynet::Expression> ret;
  for (auto & layer : fwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & layer : bwd_lstm.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
  for (auto & e : merge_input.get_params()) { ret.push_back(e); }
  for (auto & e : merge.get_params()) { ret.push_back(e); }
  for (auto & e : scorer.get_params()) { ret.push_back(e); }
  ret.push_back(empty);
  ret.push_back(fwd_guard);
  ret.push_back(bwd_guard);
  return dynet::sum(ret);
}

}