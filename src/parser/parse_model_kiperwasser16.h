#ifndef __TWPIPE_PARSER_KIPERWASSER_H__
#define __TWPIPE_PARSER_KIPERWASSER_H__

#include "parse_model.h"
#include "state.h"
#include "system.h"
#include "dynet_layer/layer.h"
#include <vector>
#include <unordered_map>

namespace twpipe {

struct Kiperwasser16Model : public ParseModel {
  struct StateCheckpointImpl : public StateCheckpoint {
    /// state machine
    ~StateCheckpointImpl() {}

    dynet::Expression f0;
    dynet::Expression f1;
    dynet::Expression f2;
    dynet::Expression f3;
  };

  struct TransitionSystemFunction {
    virtual void extract_feature(std::vector<dynet::Expression>& encoded,
                                 dynet::Expression & empty,
                                 StateCheckpointImpl & checkpoint,
                                 const State & state) = 0;
  };

  /*
  struct ArcEagerFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::Expression>& encoded,
                         dynet::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };
  */

  struct ArcStandardFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::Expression>& encoded,
                         dynet::Expression & empty,
                         StateCheckpointImpl & checkpoint,
                         const State & state) override;
  };

  struct ArcHybridFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::Expression>& encoded,
                         dynet::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State & state) override;
  };

  /*
  struct SwapFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                         dynet::expr::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };
  */

  LSTMBuilderType fwd_lstm;
  LSTMBuilderType bwd_lstm;
  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  InputLayer pretrain_emb;

  Merge3Layer merge_input;
  Merge4Layer merge;        // merge (s2, s1, s0, n0)
  DenseLayer scorer;
  std::vector<dynet::Expression> encoded;

  dynet::Parameter p_empty;
  dynet::Parameter p_fwd_guard;   // start of fwd
  dynet::Parameter p_bwd_guard;   // end of bwd
  dynet::Expression empty;
  dynet::Expression fwd_guard;
  dynet::Expression bwd_guard;

  TransitionSystemFunction* sys_func;

  unsigned size_w, dim_w, size_p, dim_p, dim_t, size_a;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  explicit Kiperwasser16Model(dynet::ParameterCollection & m,
                               unsigned size_w,  //
                               unsigned dim_w,   // word size, word dim
                               unsigned size_p,  //
                               unsigned dim_p,   // pos size, pos dim
                               unsigned dim_t,   // pword size, pword dim
                               unsigned size_a,  //
                               unsigned n_layers,
                               unsigned dim_lstm_in,
                               unsigned dim_hidden,
                               TransitionSystem& system);

  void new_graph(dynet::ComputationGraph& cg) override;

  void initialize_parser(dynet::ComputationGraph& cg,
                         const InputUnits& input,
                         StateCheckpoint * checkpoint) override;

  void perform_action(const unsigned& action,
                      dynet::ComputationGraph& cg,
                      State& state,
                      StateCheckpoint * checkpoint) override;

  StateCheckpoint * get_initial_checkpoint() override;

  StateCheckpoint * copy_checkpoint(StateCheckpoint * checkpoint) override;

  void destropy_checkpoint(StateCheckpoint * checkpoint) override;

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::Expression get_scores(StateCheckpoint * checkpoint) override;
};

}

#endif  //  end for PARSER_BILSTM_H