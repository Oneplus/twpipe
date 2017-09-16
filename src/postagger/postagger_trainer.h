#ifndef __TWPIPE_POSTAGGER_TRAINER_H__
#define __TWPIPE_POSTAGGER_TRAINER_H__

#include <iostream>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "postag_model.h"
#include "twpipe/trainer.h"
#include "twpipe/optimizer_builder.h"
#include "twpipe/ensemble.h"

namespace po = boost::program_options;

namespace twpipe {

struct PostaggerTrainer : public Trainer {
  PostagModel & engine;
  OptimizerBuilder & opt_builder;
  unsigned dim;

  PostaggerTrainer(PostagModel & engine,
                   OptimizerBuilder & opt_builder,
                   const po::variables_map & conf);

  void train(const Corpus & corpus);

  float evaluate(const Corpus & corpus);
};

struct PostaggerEnsembleTrainer : public PostaggerTrainer {
  PostaggerEnsembleTrainer(PostagModel & engine,
                           OptimizerBuilder & opt_builder,
                           const po::variables_map & conf);

  static po::options_description get_options();

  void train(Corpus & corpus, EnsembleInstances & ensemble_instances);
};

}

#endif // !__TWPIPE_POSTAGGER_TRAINER_H__
