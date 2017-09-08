#ifndef __TWPIPE_PARSER_TRAINER_H__
#define __TWPIPE_PARSER_TRAINER_H__

#include <iostream>
#include <set>
#include <boost/program_options.hpp>
#include "dynet/training.h"
#include "parse_model.h"
#include "noisify.h"
#include "twpipe/trainer.h"
#include "twpipe/optimizer_builder.h"

namespace po = boost::program_options;

namespace twpipe {

struct ParserTrainer : public Trainer {
  ParseModel & engine;
  OptimizerBuilder & opt_builder;
  std::string noisify_method_name;
  float singleton_dropout_prob;

  ParserTrainer(ParseModel & engine,
                OptimizerBuilder & opt_builder,
                const po::variables_map & conf);

  static po::options_description get_options();

  float evaluate(Corpus & corpus);
};

struct SupervisedTrainer : public ParserTrainer {
  enum ORACLE_TYPE { kStatic, kDynamic };
  enum OBJECTIVE_TYPE { kCrossEntropy, kRank, kBipartieRank, kStructure };
  ORACLE_TYPE oracle_type;
  OBJECTIVE_TYPE objective_type;
  float do_pretrain_iter;
  float do_explore_prob;
  unsigned beam_size;
  bool allow_nonprojective;

  static po::options_description get_options();

  SupervisedTrainer(ParseModel & engine,
                    OptimizerBuilder & opt_builder,
                    const po::variables_map& conf);

  /* Code for supervised pretraining. */
  void train(Corpus& corpus);

  float train_full_tree(const InputUnits& input_units,
                        const ParseUnits& parse_units,
                        dynet::Trainer* trainer,
                        unsigned iter);

  float train_structure_full_tree(const InputUnits & input_units,
                                  const ParseUnits & parse_units,
                                  dynet::Trainer * trainer,
                                  unsigned beam_size);

  void add_loss_one_step(dynet::Expression & score_expr,
                         const unsigned & best_gold_action,
                         const unsigned & worst_gold_action,
                         const unsigned & best_non_gold_action,
                         std::vector<dynet::Expression> & loss);

  void get_orders(Corpus& corpus,
                  std::vector<unsigned>& order,
                  bool non_projective);
};

struct EnsembleInstance {
  std::vector<unsigned> actions;
  std::vector<std::vector<float>> probs;

  EnsembleInstance(std::vector<unsigned> & actions,
                   std::vector<std::vector<float>> & probs);
};

typedef std::unordered_map<unsigned, EnsembleInstance> EnsembleInstances;

struct SupervisedEnsembleTrainer : public ParserTrainer {
  SupervisedEnsembleTrainer(ParseModel & engine,
                            OptimizerBuilder & opt_builder,
                            const po::variables_map & conf);

  static po::options_description get_options();

  static void load_ensemble_instances(const std::string & path,
                                      EnsembleInstances & instances);

  void train(Corpus & corpus, EnsembleInstances & ensemble_instances);

  float train_full_tree(const InputUnits & input_units,
                        const EnsembleInstance & ensemble_instance,
                        dynet::Trainer * trainer);
};


}

#endif  //  end for TRAIN_SUPERVISED_H