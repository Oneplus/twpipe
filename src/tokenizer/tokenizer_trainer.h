#ifndef __TWPIPE_TOKENIZER_TRAINER_H__
#define __TWPIPE_TOKENIZER_TRAINER_H__

#include <iostream>
#include <boost/program_options.hpp>
#include "tokenize_model.h"
#include "twpipe/trainer.h"
#include "twpipe/corpus.h"
#include "twpipe/optimizer_builder.h"

namespace po = boost::program_options;

namespace twpipe {

struct TokenizerTrainer : public Trainer {
  AbstractTokenizeModel & engine;
  OptimizerBuilder & opt_builder;
  const char* phase_name;

  TokenizerTrainer(AbstractTokenizeModel & engine,
                   OptimizerBuilder & opt_builder,
                   po::variables_map & conf);

  void train(const Corpus & corpus);

  float evaluate(const Corpus & corpus);
};


}

#endif  //  end for __TWPIPE_TOKENIZER_TRAINER_H__