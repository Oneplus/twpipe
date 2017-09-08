#include "parser_ensemble_trainer.h"
#include "tree.h"

namespace twpipe {

EnsembleInstance::EnsembleInstance(std::vector<unsigned>& actions, 
                                   std::vector<std::vector<float>>& probs) :
  actions(actions),
  probs(probs) {
}

SupervisedEnsembleTrainer::SupervisedEnsembleTrainer(ParseModel & engine,
                                                     OptimizerBuilder & opt_builder,
                                                     const po::variables_map & conf) :
  Trainer(conf),
  engine(engine),
  opt_builder(opt_builder) {
}

void SupervisedEnsembleTrainer::train(Corpus & corpus,
                                      EnsembleInstances & ensemble_instances) {
  dynet::ParameterCollection & model = engine.model;
  dynet::Trainer * trainer = opt_builder.build(model);

  std::vector<unsigned> order;
  bool allow_nonprojective = engine.sys.allow_nonprojective();
  for (auto & payload : ensemble_instances) { order.push_back(payload.first); }

  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits & units = corpus.training_data.at(sid).input_units;
      const EnsembleInstance & inst = ensemble_instances.at(sid);
      
      train_full_tree(units, inst, trainer);
    }
  }
}

}
