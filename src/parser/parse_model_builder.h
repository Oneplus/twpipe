#ifndef __TWPIPE_PARSER_PARSE_MODEL_BUILDER_H__
#define __TWPIPE_PARSER_PARSE_MODEL_BUILDER_H__

#include <iostream>
#include "parse_model.h"
#include "dynet/model.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct ParseModelBuilder {
  TransitionSystem * system;
  std::string system_name;
  std::string arch_name;

  unsigned char_dim;
  unsigned word_dim;
  unsigned pos_dim;
  unsigned pretrained_dim;
  unsigned n_actions;
  unsigned action_dim;
  unsigned label_dim;
  unsigned n_layers;
  unsigned lstm_input_dim;
  unsigned hidden_dim;

  ParseModelBuilder(po::variables_map & conf);

  ParseModel* build(dynet::ParameterCollection & model);

  void to_json();

  ParseModel * from_json(dynet::ParameterCollection & model);
};

}

#endif  //  end for PARSER_BUILDER_H