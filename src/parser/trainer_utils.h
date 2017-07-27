#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "corpus.h"
#include "dynet/model.h"
#include "dynet/training.h"

namespace po = boost::program_options;

void get_orders(Corpus& corpus,
                std::vector<unsigned>& order,
                bool non_projective);

po::options_description get_optimizer_options();

dynet::Trainer* get_trainer(const po::variables_map& conf,
                            dynet::Model& model);

void update_trainer(const po::variables_map& conf,
                    dynet::Trainer* trainer);

std::string get_model_name(const po::variables_map& conf,
                           const std::string& prefix);

#endif  //  end for TRAIN_H