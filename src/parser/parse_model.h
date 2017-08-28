#ifndef __TWPIPE_PARSER_PARSE_MODEL_H__
#define __TWPIPE_PARSER_PARSE_MODEL_H__

#include "state.h"
#include "system.h"
#include "twpipe/layer.h"
#include "twpipe/corpus.h"
#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct ParseModel {
  static po::options_description get_options();

  struct StateCheckpoint {
    virtual ~StateCheckpoint() {}
  };

  dynet::ParameterCollection & model;
  TransitionSystem & sys;

  ParseModel(dynet::ParameterCollection & m, TransitionSystem& s);

  void predict(const std::vector<std::string> & words,
               const std::vector<std::string> & postags,
               std::vector<unsigned> & heads,
               std::vector<std::string> & deprels);

  void label(const std::vector<std::string> & words,
             const std::vector<std::string> & postags,
             const std::vector<unsigned> & heads,
             std::vector<std::string> & deprels);

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
  virtual dynet::Expression get_scores(StateCheckpoint * checkpoint) = 0;
 
  virtual void raw_to_input_units(const std::vector<std::string> & words,
                                  const std::vector<std::string> & postags,
                                  InputUnits & units) = 0;

  void parse_units_to_raw(const ParseUnits & units,
                          std::vector<unsigned> & heads,
                          std::vector<std::string> & deprels,
                          bool add_pseduo_root=false);

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

}

#endif  //  end for __TWPIPE_PARSER_PARSE_MODEL_H__
