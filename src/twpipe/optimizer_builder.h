#ifndef __TWPIPE_OPTIMIZER_BUILDER_H__
#define __TWPIPE_OPTIMIZER_BUILDER_H__

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/model.h"
#include "dynet/training.h"

namespace po = boost::program_options;

namespace twpipe {

struct OptimizerBuilder {
  enum OptimizerType {
    kSimpleSGD,
    kMomentumSGD,
    kAdaGrad,
    kAdaDelta,
    kRMSProp,
    kAdam
  };

  OptimizerType optimizer_type;
  float eta0;
  float adam_beta1;
  float adam_beta2;
  bool enable_clipping;

  static po::options_description get_options();

  OptimizerBuilder(const po::variables_map& conf);

  dynet::Trainer * build(dynet::ParameterCollection & model);
};

}

#endif  //  end for TRAIN_H