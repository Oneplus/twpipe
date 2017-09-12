#include "postagger_trainer.h"
#include "twpipe/model.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/embedding.h"

namespace twpipe {

PostaggerTrainer::PostaggerTrainer(PostagModel & engine, 
                                   OptimizerBuilder & opt_builder,
                                   po::variables_map & conf) :
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
 
    float acc = n_recall / n_total;
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

}
