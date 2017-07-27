#ifndef PARSER_DYER15_H
#define PARSER_DYER15_H

#include "parser.h"
#include "layer.h"
#include "corpus.h"
#include "state.h"
#include "system.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserDyer15 : public Parser {
  struct StateCheckpointImpl : public StateCheckpoint {
    /// state machine
    ~StateCheckpointImpl() {}

    dynet::RNNPointer s_pointer;
    dynet::RNNPointer q_pointer;
    dynet::RNNPointer a_pointer;
    std::vector<dynet::expr::Expression> stack;
    std::vector<dynet::expr::Expression> buffer;
  };

  struct TransitionSystemFunction {
    virtual void perform_action(const unsigned& action,
                                dynet::ComputationGraph& cg,
                                dynet::LSTMBuilder& a_lstm,
                                dynet::LSTMBuilder& s_lstm,
                                dynet::LSTMBuilder& q_lstm,
                                Merge3Layer& composer,
                                StateCheckpointImpl & checkpoint,
                                dynet::expr::Expression& act_expr,
                                dynet::expr::Expression& rel_expr) = 0;
  };

  struct ArcEagerFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        dynet::LSTMBuilder& a_lstm,
                        dynet::LSTMBuilder& s_lstm,
                        dynet::LSTMBuilder& q_lstm,
                        Merge3Layer& composer,                  
                        StateCheckpointImpl & checkpoint,
                        dynet::expr::Expression& act_expr,
                        dynet::expr::Expression& rel_expr) override;

  };

  struct ArcStandardFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        dynet::LSTMBuilder& a_lstm, 
                        dynet::LSTMBuilder& s_lstm,
                        dynet::LSTMBuilder& q_lstm,
                        Merge3Layer& composer,
                        StateCheckpointImpl & checkpoint,
                        dynet::expr::Expression& act_expr,
                        dynet::expr::Expression& rel_expr) override;
  };
  
  struct ArcHybridFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        dynet::LSTMBuilder& a_lstm,
                        dynet::LSTMBuilder& s_lstm,
                        dynet::LSTMBuilder& q_lstm,
                        Merge3Layer& composer,
                        StateCheckpointImpl & checkpoint,
                        dynet::expr::Expression& act_expr,
                        dynet::expr::Expression& rel_expr) override;
  };

  struct SwapFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        dynet::LSTMBuilder& a_lstm,
                        dynet::LSTMBuilder& s_lstm,
                        dynet::LSTMBuilder& q_lstm,
                        Merge3Layer& composer,
                        StateCheckpointImpl & checkpoint,
                        dynet::expr::Expression& act_expr,
                        dynet::expr::Expression& rel_expr) override;
  };

  dynet::LSTMBuilder s_lstm;
  dynet::LSTMBuilder q_lstm;
  dynet::LSTMBuilder a_lstm;

  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;
  SymbolEmbedding act_emb;
  SymbolEmbedding rel_emb;

  Merge3Layer merge_input;  // merge (word, pos, preword)
  Merge3Layer merge;        // merge (s_lstm, q_lstm, a_lstm)
  Merge3Layer composer;     // compose (head, modifier, relation)
  DenseLayer scorer;

  dynet::Parameter p_action_start;  // start of action
  dynet::Parameter p_buffer_guard;  // end of buffer
  dynet::Parameter p_stack_guard;   // end of stack
  dynet::expr::Expression action_start;
  dynet::expr::Expression buffer_guard;
  dynet::expr::Expression stack_guard;

  /// The reference
  TransitionSystemFunction* sys_func;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  /// The Configurations: useful for other models.
  unsigned size_w, dim_w, size_p, dim_p, size_t, dim_t, size_a, dim_a, dim_l;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  explicit ParserDyer15(dynet::Model& m,
                          unsigned size_w,  //
                          unsigned dim_w,   // word size, word dim
                          unsigned size_p,  //
                          unsigned dim_p,   // pos size, pos dim
                          unsigned size_t,  //
                          unsigned dim_t,   // pword size, pword dim
                          unsigned size_a,  //
                          unsigned dim_a,   // act size, act dim
                          unsigned dim_l,
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

#endif  //  end for PARSER_H
