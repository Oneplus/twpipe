#ifndef __TOKENIZER_TOKENIZE_MODEL_H__
#define __TOKENIZER_TOKENIZE_MODEL_H__

#include <boost/program_options.hpp>
#include <tuple>
#include "twpipe/corpus.h"
#include "twpipe/model.h"
#include "dynet/expr.h"

namespace po = boost::program_options;

namespace twpipe {

struct TokenizeModel {
  static po::options_description get_options();

  dynet::ParameterCollection & model;
  const Alphabet & char_map;
  unsigned space_cid;

  TokenizeModel(dynet::ParameterCollection & model,
                const Alphabet & char_map);

  virtual void new_graph(dynet::ComputationGraph & cg) = 0;

  virtual void decode(const std::string & input, std::vector<std::string> & result) = 0;

  virtual dynet::Expression objective(const Instance & inst) = 0;
 
  void tokenize(const std::string & input);
  
  void tokenize(const std::string & input, std::vector<std::string> & result);

  std::tuple<float, float, float> evaluate(const std::vector<std::string> & gold,
                                           const std::vector<std::string> & prediction);
};

}

#endif  //  end for __TOKENIZER_TOKENIZE_MODEL_H__