#ifndef PARSER_NO_HISTORY_H
#define PARSER_NO_HISTORY_H

#include "parser.h"
#include "layer.h"
#include "corpus.h"
#include "state.h"
#include "system.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserArchNoHistory : public Parser {
  struct StateCheckpointImpl : public StateCheckpoint {
    ~StateCheckpointImpl() {}
    
    /// state machine
    dynet::RNNPointer s_pointer;
    dynet::RNNPointer q_pointer;
    std::vector<dynet::expr::Expression> stack;
    std::vector<dynet::expr::Expression> buffer;
  };

  struct TransitionSystemFunction {
    virtual void perform_action(const unsigned& action,
                                dynet::ComputationGraph& cg,
                                std::vector<dynet::expr::Expression>& stack,
                                std::vector<dynet::expr::Expression>& buffer,
                                dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                                dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                                Merge3Layer& composer,
                                dynet::expr::Expression& rel_expr) = 0;
  };

  struct ArcEagerFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        std::vector<dynet::expr::Expression>& stack,
                        std::vector<dynet::expr::Expression>& buffer,
                        dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                        dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                        Merge3Layer& composer,
                        dynet::expr::Expression& rel_expr) override;
  };

  struct ArcStandardFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        std::vector<dynet::expr::Expression>& stack,
                        std::vector<dynet::expr::Expression>& buffer,
                        dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                        dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                        Merge3Layer& composer,
                        dynet::expr::Expression& rel_expr) override;
  };

  struct ArcHybridFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        std::vector<dynet::expr::Expression>& stack,
                        std::vector<dynet::expr::Expression>& buffer,
                        dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                        dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                        Merge3Layer& composer,
                        dynet::expr::Expression& rel_expr) override;
  };

  struct SwapFunction : public TransitionSystemFunction {
    void perform_action(const unsigned& action,
                        dynet::ComputationGraph& cg,
                        std::vector<dynet::expr::Expression>& stack,
                        std::vector<dynet::expr::Expression>& buffer,
                        dynet::LSTMBuilder& s_lstm, dynet::RNNPointer& s_pointer,
                        dynet::LSTMBuilder& q_lstm, dynet::RNNPointer& q_pointer,
                        Merge3Layer& composer,
                        dynet::expr::Expression& rel_expr) override;
  };

  dynet::LSTMBuilder s_lstm;
  dynet::LSTMBuilder q_lstm;
  SymbolEmbedding word_emb;
  SymbolEmbedding pos_emb;
  SymbolEmbedding preword_emb;
  SymbolEmbedding rel_emb;

  Merge3Layer merge_input;  // merge (word, pos, preword)
  Merge2Layer merge;        // merge (s_lstm, q_lstm, a_lstm)
  Merge3Layer composer;     // compose (head, modifier, relation)
  DenseLayer scorer;

  dynet::Parameter p_buffer_guard;  // end of buffer
  dynet::Parameter p_stack_guard;   // end of stack
  dynet::expr::Expression buffer_guard;
  dynet::expr::Expression stack_guard;

  /// The reference
  TransitionSystemFunction* sys_func;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  /// The Configurations: useful for other models.
  unsigned size_w, dim_w, size_p, dim_p, size_t, dim_t, size_l, dim_l;
  unsigned n_layers, dim_lstm_in, dim_hidden;

  explicit ParserArchNoHistory(dynet::Model& m,
                               unsigned size_w,  //
                               unsigned dim_w,   // word size, word dim
                               unsigned size_p,  //
                               unsigned dim_p,   // pos size, pos dim
                               unsigned size_t,  //
                               unsigned dim_t,   // pword size, pword dim
                               unsigned size_l,  //
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

  StateCheckpoint * get_initial_checkpoint();

  StateCheckpoint * copy_checkpoint(StateCheckpoint * checkpoint);

  void destropy_checkpoint(StateCheckpoint * checkpoint);

  void perform_action(const unsigned& action,
                      dynet::ComputationGraph& cg,
                      State& state,
                      StateCheckpoint * checkpoint) override;

  /// Get the un-softmaxed scores from the LSTM-parser.
  dynet::expr::Expression get_scores(StateCheckpoint * checkpoint) override;
};

#endif  //  end for PARSER_H
