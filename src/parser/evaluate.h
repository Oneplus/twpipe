#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <set>
#include "corpus.h"
#include "parser.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

float evaluate(const po::variables_map& conf,
               Corpus& corpus,
               Parser& parser,
               const std::string& output,
               bool labelling_only = false);

float beam_search(const po::variables_map& conf,
                  Corpus& corpus,
                  Parser& parser,
                  const std::string& output);

#endif  //  end for EVALUATE_H