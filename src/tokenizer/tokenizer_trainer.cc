#include <fstream>
#include "tokenizer_trainer.h"
#include "twpipe/logging.h"

namespace twpipe {

TokenizerTrainer::TokenizerTrainer(TokenizeModel & engine,
                                   OptimizerBuilder & opt_builder,
                                   po::variables_map & conf) :
  Trainer(conf),
  opt_builder(opt_builder),
  engine(engine) {
}

void twpipe::TokenizerTrainer::train(const Corpus & corpus) {
  _INFO << "[tokenize|train] training tokenizer model";
  _INFO << "[tokenize|train] size of dataset = " << corpus.n_train;

  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  _INFO << "[tokenize|train] going to train " << max_iter << " iterations";

  dynet::Trainer * trainer = opt_builder.build(engine.model);

  float eta0 = trainer->learning_rate;
  float best_f = 0.f;
  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    std::shuffle(order.begin(), order.end(), *dynet::rndeng);
    _INFO << "[tokenize|train] start training at " << iter << "-th iteration.";

    float loss = 0;
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
    _INFO << "[tokenize|train] loss = " << loss;

    float n_recall = 0, n_pred = 0, n_gold = 0;
    for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
      const Instance & inst = corpus.devel_data.at(sid);

      dynet::ComputationGraph cg;
      engine.new_graph(cg);
      std::vector<std::string> result;
      engine.decode(inst.raw_sentence, result);

      std::vector<std::string> gold;
      for (unsigned i = 1; i < inst.input_units.size(); ++i) {
        gold.push_back(inst.input_units[i].word);
      }
      auto payload = engine.evaluate(gold, result);
      n_recall += std::get<0>(payload);
      n_pred += std::get<1>(payload);
      n_gold += std::get<2>(payload);
    }

    float f = 2 * n_recall / (n_pred + n_gold);
    _INFO << "[tokenize|train] iteration " << iter << ", f-score = " << f;

    if (f > best_f) {
      best_f = f;
      _INFO << "[tokenize|train] new record achieved " << best_f << ", model saved.";
      Model::get()->to_json(Model::kTokenizerName, engine.model);
    }
    trainer->learning_rate = eta0 / (1. + static_cast<float>(iter) * 0.08);
  }

  delete trainer;
}

}