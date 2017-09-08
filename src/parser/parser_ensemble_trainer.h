#ifndef __TWPIPE_PARSER_ENSEMBLE_TRAINER_H__
#define __TWPIPE_PARSER_ENSEMBLE_TRAINER_H__

#include <iostream>
#include <boost/program_options.hpp>
#include "parse_model.h"
#include "twpipe/trainer.h"
#include "twpipe/corpus.h"
#include "twpipe/optimizer_builder.h"

namespace twpipe {

struct EnsembleInstance {
  std::vector<unsigned> actions;
  std::vector<std::vector<float>> probs;

  EnsembleInstance(std::vector<unsigned> & actions,
                   std::vector<std::vector<float>> & probs);
};

typedef std::unordered_map<unsigned, EnsembleInstance> EnsembleInstances;

struct SupervisedEnsembleTrainer : public Trainer {
  ParseModel & engine;
  OptimizerBuilder & opt_builder;

  SupervisedEnsembleTrainer(ParseModel & engine,
                            OptimizerBuilder & opt_builder,
                            const po::variables_map & conf);

  void train(Corpus & corpus, EnsembleInstances & ensemble_instances);

  void train_full_tree(const InputUnits & input_units,
                       const EnsembleInstance & ensemble_instance,
                       dynet::Trainer * trainer);
};

}

#endif  //  end for __TWPIPE_PARSER_ENSEMBLE_TRAINER_H__
