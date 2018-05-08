#include <fstream>
#include "tokenizer_trainer.h"
#include "twpipe/logging.h"

namespace twpipe {

TokenizerTrainer::TokenizerTrainer(AbstractTokenizeModel & engine,
                                   OptimizerBuilder & opt_builder,
                                   po::variables_map & conf) :
  Trainer(conf),
  engine(engine),
  opt_builder(opt_builder) {
  if (conf["train-segmentor-and-tokenizer"].as<bool>()) {
    phase_name = Model::kSentenceSegmentAndTokenizeName;
  } else {
    phase_name = Model::kTokenizerName;
  }
}

float twpipe::TokenizerTrainer::evaluate(const Corpus & corpus) {
  float n_recall = 0, n_pred = 0, n_gold = 0;
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    const Instance & inst = corpus.devel_data.at(sid);

    auto payload = engine.evaluate(inst);
    n_recall += std::get<0>(payload);
    n_pred += std::get<1>(payload);
    n_gold += std::get<2>(payload);
  }
  float p = n_recall / n_gold;
  float r = n_recall / n_pred;
  float f = 2 * p * r / (p + r);
  return f;
}

void twpipe::TokenizerTrainer::train(const Corpus & corpus) {
  _INFO << "[tokenize|train] training " << phase_name << " model";
  _INFO << "[tokenize|train] size of dataset = " << corpus.n_train;

  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  _INFO << "[tokenize|train] going to train " << max_iter << " iterations";

  dynet::Trainer * trainer = opt_builder.build(engine.model);

  float best_f = 0.f;
  unsigned n_processed = 0;

  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    std::shuffle(order.begin(), order.end(), *dynet::rndeng);
    _INFO << "[tokenize|train] start training at " << iter << "-th iteration.";

    float loss = 0;
    for (unsigned sid = 0; sid < corpus.n_train; ++sid) {
      const Instance & inst = corpus.training_data.at(order[sid]);
      {
        dynet::ComputationGraph cg;
        engine.new_graph(cg);
        dynet::Expression loss_expr = engine.objective(inst);
        if (lambda_ > 0) {
          loss_expr = loss_expr + (0.5f * lambda_ * inst.input_units.size()) * engine.l2();
        }
        float l = dynet::as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        loss += l;

        trainer->update();
        n_processed++;
      }
      if (need_evaluate(iter, n_processed)) {
        float f = evaluate(corpus);
        float prop = static_cast<float>(n_processed) / order.size();
        if (f > best_f) {
          _INFO << "[tokenize|train] " << prop << "% trained, fscore on heldout = " << f
                << ", new best achieved, saved.";
          best_f = f;
          Model::get()->to_json(phase_name, engine.model);
        } else {
          _INFO << "[tokenize|train] " << prop << "% trained, fscore on heldout = " << f;
        }
      }
    }
    _INFO << "[tokenize|train] end of iter #" << iter << ", loss=" << loss;
    if (need_evaluate(iter)) {
      float f = evaluate(corpus);
      if (f > best_f) {
        _INFO << "[tokenize|train] end of iter #" << iter << ", fscore on heldout = " << f
              << ", new best achieved, saved.";
        best_f = f;
        Model::get()->to_json(phase_name, engine.model);
      } else {
        _INFO << "[tokenize|train] end of iter #" << iter << ", fscore on heldout = " << f;
      }
    }
    opt_builder.update(trainer, iter);
  }
  _INFO << "[tokenize|train] training is done, best fscore is: " << best_f;
  delete trainer;
}

}