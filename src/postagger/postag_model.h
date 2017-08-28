#ifndef __TWPIPE_POSTAG_MODEL_H__
#define __TWPIPE_POSTAG_MODEL_H__

#include <boost/program_options.hpp>
#include "twpipe/corpus.h"
#include "dynet/expr.h"

namespace po = boost::program_options;

namespace twpipe {

struct PostagModel {
  static po::options_description get_options();

  dynet::ParameterCollection & model;
  unsigned pos_size;

  PostagModel(dynet::ParameterCollection & model);

  virtual void new_graph(dynet::ComputationGraph & cg) = 0;

  virtual void decode(const std::vector<std::string> & words,
                      std::vector<std::string> & tags) = 0;

  virtual dynet::Expression objective(const Instance & inst) = 0;

  void postag(const std::vector<std::string> & words);

  void postag(const std::vector<std::string> & words,
              std::vector<std::string> & tags);

  std::pair<float, float> evaluate(const std::vector<std::string> & gold,
                                   const std::vector<std::string> & prediction);
};

}

#endif  //  end for __TWPIPE_POSTAG_MODEL_H__