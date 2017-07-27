#ifndef PARSER_BUILDER_H
#define PARSER_BUILDER_H

#include <iostream>
#include "parser.h"
#include "dynet/model.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct ParserBuilder {
  static po::options_description get_options();
  static Parser* build(const po::variables_map& conf,
                       dynet::Model& model,
                       TransitionSystem& sys,
                       const Corpus& corpus,
                       const std::unordered_map<unsigned, std::vector<float>>& pretrained);
};
#endif  //  end for PARSER_BUILDER_H