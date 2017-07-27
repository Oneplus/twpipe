#include "postagger_trainer.h"
#include "twpipe/model.h"
#include "twpipe/logging.h"

namespace twpipe {

PostaggerTrainer::PostaggerTrainer(PostagModel & engine, 
                                   OptimizerBuilder & opt_builder,
                                   const StrEmbeddingType & embeddings,
                                   po::variables_map & conf) :
  Trainer(conf),
  opt_builder(opt_builder),
  engine(engine),
  embeddings(embeddings) {
  dim = embeddings.at(Corpus::BAD0).size();
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
      std::vector<std::vector<float>> values;
      for (unsigned i = 1; i < inst.input_units.size(); ++i) {
        auto it = embeddings.find(inst.input_units[i].norm_word);
        values.push_back(it == embeddings.end() ? std::vector<float>(dim, 0.) : it->second);
      }
      dynet::Expression loss_expr = engine.objective(inst, values);
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
      std::vector<std::string> words;
      std::vector<std::string> result;
      std::vector<std::vector<float>> values;
      for (unsigned i = 1; i < inst.input_units.size(); ++i) {
        auto it = embeddings.find(inst.input_units[i].norm_word);
        words.push_back(inst.input_units[i].word);
        values.push_back(it == embeddings.end() ? std::vector<float>(dim, 0.) : it->second);
      }
      engine.decode(words, values, result);
      auto payload = engine.evaluate(words, result);

      n_recall += payload.first;
      n_total += payload.second;

      float acc = n_recall / n_total;
      _INFO << "[postag|train] iteration " << iter << ", accuracy = " << acc;
    
      if (acc > best_acc) {
        best_acc = acc;
        _INFO << "[postag|train] new record achieved " << best_acc << ", model saved.";
        Model::get()->to_json(Model::kPostaggerName, engine.model);
      }
      trainer->learning_rate = eta0 / (1. + static_cast<float>(iter) * 0.08);
    }
  }

  delete trainer;
}

}