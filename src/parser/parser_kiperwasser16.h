#ifndef PARSER_KIPERWASSER_H
#define PARSER_KIPERWASSER_H

#include "parser.h"
#include "layer.h"
#include "corpus.h"
#include "state.h"
#include "system.h"
#include <vector>
#include <unordered_map>

struct ParserKiperwasser16 : public Parser {
  struct StateCheckpointImpl : public StateCheckpoint {
    /// state machine
    ~StateCheckpointImpl() {}

    dynet::expr::Expression f0;
    dynet::expr::Expression f1;
    dynet::expr::Expression f2;
    dynet::expr::Expression f3;
  };

  struct TransitionSystemFunction {
    virtual void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                                 dynet::expr::Expression& empty,
                                 StateCheckpointImpl & checkpoint,
                                 const State& state) = 0;
  };

  struct ArcEagerFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                         dynet::expr::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };

  struct ArcStandardFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                         dynet::expr::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };

  struct ArcHybridFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                         dynet::expr::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };

  struct SwapFunction : public TransitionSystemFunction {
    void extract_feature(std::vector<dynet::expr::Expression>& encoded,
                         dynet::expr::Expression& empty,
                         StateCheckpointImpl & checkpoint,
                         const State& state) override;
  };

  dynet::LSTMBuilder fwd_lstm;
  dynet::LSTMBuilder bwd_lstm;
  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;
  
  Merge3Layer merge_input; 
  Merge4Layer merge;        // merge (s2, s1, s0, n0)
  DenseLayer scorer;
  std::vector<dynet::expr::Expression> encoded;

  dynet::Parameter p_empty;
  dynet::Parameter p_fwd_guard;   // start of fwd
  dynet::Parameter p_bwd_guard;   // end of bwd
  dynet::expr::Expression empty;
  dynet::expr::Expression fwd_guard;
  dynet::expr::Expression bwd_guard;
  
  TransitionSystemFunction* sys_func;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;
  
  unsigned size_w, dim_w, size_p, dim_p, size_t, dim_t, size_a;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  explicit ParserKiperwasser16(dynet::Model& m,
                               unsigned size_w,  //
                               unsigned dim_w,   // word size, word dim
                               unsigned size_p,  //
                               unsigned dim_p,   // pos size, pos dim
                               unsigned size_t,  //
                               unsigned dim_t,   // pword size, pword dim
                               unsigned size_a,  //
                               unsigned n_layers,
                               unsigned dim_lstm_in,
                               unsigned dim_hidden,
                               const std::string& system_name,
                               TransitionSystem& system,
                               const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(dynet::ComputationGraph& cg) override;

  void initialize_parser(dynet::ComputationGraph& cg,
                         const InputUnits& input,
                         StateCheckpoint * checkpoint) override;

  void perform_action(const unsigned& action,
                      dynet::ComputationGraph& cg,
                      State& state,
                      StateCheckpoint * checkpoint) override;

  StateCheckpoint * get_initial_checkpoint();

  StateCheckpoint * copy_checkpoint(StateCheckpoint * checkpoint);

  void destropy_checkpoint(StateCheckpoint * checkpoint);

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::expr::Expression get_scores(StateCheckpoint * checkpoint) override;
};

#endif  //  end for PARSER_BILSTM_H