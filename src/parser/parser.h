#ifndef PARSER_H
#define PARSER_H

#include "layer.h"
#include "corpus.h"
#include "state.h"
#include "system.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct Parser {
  struct StateCheckpoint {
    virtual ~StateCheckpoint() {}
  };

  dynet::Model& model;
  TransitionSystem& sys;
  std::string system_name;

  Parser(dynet::Model& m,
         TransitionSystem& s,
         const std::string& sys_name) : 
    model(m), sys(s), system_name(sys_name) {}

  virtual void new_graph(dynet::ComputationGraph& cg) = 0;

  void initialize(dynet::ComputationGraph& cg,
                  const InputUnits& input,
                  State& state,
                  StateCheckpoint * checkpoint);

  void initialize_state(const InputUnits& input,
                        State& state);

  virtual void initialize_parser(dynet::ComputationGraph& cg,
                                 const InputUnits& input,
                                 StateCheckpoint * checkpoint) = 0;

  virtual void perform_action(const unsigned& action,
                              dynet::ComputationGraph& cg,
                              State& state,
                              StateCheckpoint * checkpoint) = 0;

  static std::pair<unsigned, float> get_best_action(const std::vector<float>& scores,
                                                    const std::vector<unsigned>& valid_actions);

  virtual StateCheckpoint * get_initial_checkpoint() = 0;

  virtual StateCheckpoint * copy_checkpoint(StateCheckpoint * checkpoint) = 0;

  virtual void destropy_checkpoint(StateCheckpoint * checkpoint) = 0;

  /// Get the un-softmaxed scores from the LSTM-parser.
  virtual dynet::expr::Expression get_scores(StateCheckpoint * checkpoint) = 0;

  void predict(dynet::ComputationGraph& cg,
               const InputUnits& input,
               ParseUnits& parse);

  void label(dynet::ComputationGraph& cg,
             const InputUnits& input,
             const ParseUnits& parse,
             ParseUnits & output);

  void beam_search(dynet::ComputationGraph& cg,
                   const InputUnits& input,
                   const unsigned& beam_size,
                   bool structure_score,
                   std::vector<ParseUnits>& parse);
};


#endif  //  end for PARSER_H
