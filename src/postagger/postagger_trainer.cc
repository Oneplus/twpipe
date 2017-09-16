#include "postagger_trainer.h"
#include "twpipe/model.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/embedding.h"

namespace twpipe {

PostaggerTrainer::PostaggerTrainer(PostagModel & engine, 
                                   OptimizerBuilder & opt_builder,
                                   const po::variables_map & conf) :
  Trainer(conf),
  engine(engine),
  opt_builder(opt_builder) {
  dim = WordEmbedding::get()->dim();
}

void PostaggerTrainer::train(const Corpus & corpus) {
  _INFO << "[postag|train] training postagger model.";
  _INFO << "[postag|train] size of dataset = " << corpus.n_train;

  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  _INFO << "[postag|train] going to train " << max_iter << " iterations";

  dynet::Trainer * trainer = opt_builder.build(engine.model);

  float eta0 = trainer->learning_rate;
  float best_acc = 0.f;

  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    std::shuffle(order.begin(), order.end(), *dynet::rndeng);
    _INFO << "[postag|train] start training at " << iter << "-th iteration.";

    float loss = 0.;
    for (unsigned sid = 0; sid < corpus.n_train; ++sid) {
      const Instance & inst = corpus.training_data.at(order[sid]);

      dynet::ComputationGraph cg;
      engine.new_graph(cg);

      dynet::Expression loss_expr = engine.objective(inst);
      float l = dynet::as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      loss += l;

      trainer->update();
    }
    _INFO << "[postag|train] loss = " << loss;

    float acc = evaluate(corpus);
    _INFO << "[postag|train] iteration " << iter << ", accuracy = " << acc;

    if (acc > best_acc) {
      best_acc = acc;
      _INFO << "[postag|train] new record achieved " << best_acc << ", model saved.";
      Model::get()->to_json(Model::kPostaggerName, engine.model);
    }
    trainer->learning_rate = eta0 / (1. + static_cast<float>(iter) * 0.08);
  }

  _INFO << "[postag|train] training is done, best accuracy is: " << best_acc;
  delete trainer;
}

float PostaggerTrainer::evaluate(const Corpus & corpus) {
  float n_recall = 0, n_total = 0;
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    const Instance & inst = corpus.devel_data.at(sid);

    dynet::ComputationGraph cg;
    engine.new_graph(cg);

    unsigned len = inst.input_units.size();
    std::vector<std::string> words(len - 1);
    std::vector<std::string> gold_postags(len - 1), pred_postags;
    std::vector<std::vector<float>> values;
    for (unsigned i = 1; i < inst.input_units.size(); ++i) {
      words[i - 1] = inst.input_units[i].word;
      gold_postags[i - 1] = inst.input_units[i].postag;
    }
    engine.decode(words, pred_postags);
    auto payload = engine.evaluate(gold_postags, pred_postags);

    n_recall += payload.first;
    n_total += payload.second;
  }

  return n_recall / n_total;
}

PostaggerEnsembleTrainer::PostaggerEnsembleTrainer(PostagModel & engine, 
                                                   OptimizerBuilder & opt_builder,
                                                   const po::variables_map & conf) :
  PostaggerTrainer(engine, opt_builder, conf) {
}

po::options_description PostaggerEnsembleTrainer::get_options() {
  po::options_description cmd("Postagger ensemble learning options");
  cmd.add_options()
    ("postag-ensemble-data", po::value<std::string>(), "The path to the ensemble data.")
    ;
  return cmd;
}

void PostaggerEnsembleTrainer::train(Corpus & corpus,
                                     EnsembleInstances & ensemble_instances) {
  _INFO << "[postag|ensemble|train] start lstm-parser supervised training.";

  dynet::ParameterCollection & model = engine.model;
  dynet::Trainer * trainer = opt_builder.build(model);
  float eta0 = trainer->learning_rate;

  std::vector<unsigned> order;
  for (auto & payload : ensemble_instances) {
    unsigned id = payload.first;
    if (ensemble_instances.count(id) == 0) { continue; }
    order.push_back(id);
  }

  float llh = 0.f;
  float best_acc = -1.f;

  _INFO << "[postag|ensemble|train] will stop after " << max_iter << " iterations.";
  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    llh = 0.f;
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits & units = corpus.training_data.at(sid).input_units;
      const EnsembleInstance & inst = ensemble_instances.at(sid);

      dynet::ComputationGraph cg;
      engine.new_graph(cg);

      unsigned n_words = units.size() - 1;
      std::vector<std::string> words(n_words);
      for (unsigned i = 1; i < units.size(); ++i) {
        words[i - 1] = units[i].word;
      }
      engine.initialize(words);
      unsigned prev_label = AlphabetCollection::get()->pos_map.get(Corpus::ROOT);

      std::vector<dynet::Expression> losses;
      const std::vector<unsigned> & actions = inst.categories;
      const std::vector<std::vector<float>> & probs = inst.probs;
      for (unsigned i = 0; i < n_words; ++i) {
        dynet::Expression feature = engine.get_feature(i, prev_label);
        dynet::Expression logits = engine.get_emit_score(feature);
        const std::vector<float> & prob = probs.at(i);
        unsigned dim = prob.size();
        losses.push_back(dynet::dot_product(dynet::input(cg, { dim }, prob), logits));
        prev_label = actions.at(i);
      }
      dynet::Expression loss_expr = dynet::sum(losses);
      float l = dynet::as_scalar(cg.forward(loss_expr));
      cg.backward(loss_expr);
      llh += l;
    }
    _INFO << "[postag|ensemble|train] end of iter #" << iter << " loss " << llh;
    float acc = evaluate(corpus);
    if (acc > best_acc) {
      best_acc = acc;
      _INFO << "[postag|ensemble|train] new best record achieved: " << best_acc << ", saved.";
      Model::get()->to_json(Model::kPostaggerName, engine.model);
    }
    trainer->learning_rate = eta0 / (1. + static_cast<float>(iter) * 0.08);
  }
}

}
