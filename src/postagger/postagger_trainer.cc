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
  float best_acc = 0.f;
  unsigned n_processed = 0;

  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    std::shuffle(order.begin(), order.end(), *dynet::rndeng);
    _INFO << "[postag|train] start training at " << iter << "-th iteration.";

    float loss = 0.f;
    for (unsigned sid : order) {
      const Instance & inst = corpus.training_data.at(sid);

      {
        dynet::ComputationGraph cg;
        engine.new_graph(cg);
        dynet::Expression loss_expr = engine.objective(inst);
        if (lambda_ > 0) { loss_expr = loss_expr + lambda_ * engine.l2(); }
        float l = dynet::as_scalar(cg.forward(loss_expr));
        cg.backward(loss_expr);
        loss += l;
        trainer->update();
        n_processed++;
      }
      if (need_evaluate(iter, n_processed)) {
        float acc = evaluate(corpus);
        if (acc > best_acc) {
          float prop = static_cast<float>(n_processed) / order.size();
          if (acc > best_acc) {
            _INFO << "[postag|train] " << prop << "% trained, ACC on heldout = " << acc 
              << ", new best achieved, saved.";
            best_acc = acc;
            Model::get()->to_json(Model::kPostaggerName, engine.model);
          } else {
            _INFO << "[postag|train] " << prop << "% trained, ACC on heldout = " << acc;
          }
        }
      }
    }
    _INFO << "[postag|train] end of iter #" << iter << ", loss = " << loss;
    if (need_evaluate(iter)) {
      float acc = evaluate(corpus);
      if (acc > best_acc) {
        best_acc = acc;
        _INFO << "[postag|train] end of iter #" << iter << ", ACC on heldout = " << acc
          << ", new best achieved, saved.";
        Model::get()->to_json(Model::kPostaggerName, engine.model);
      } else {
        _INFO << "[postag|train] end of iter #" << iter << ", ACC on heldout = " << acc;
      }
    }
    opt_builder.update(trainer, iter);
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
    ("pos-ensemble-data", po::value<std::string>(), "The path to the ensemble data.")
    ;
  return cmd;
}

void PostaggerEnsembleTrainer::train(Corpus & corpus,
                                     EnsembleInstances & ensemble_instances) {
  _INFO << "[postag|ensemble|train] start postagger supervised training.";

  dynet::ParameterCollection & model = engine.model;
  dynet::Trainer * trainer = opt_builder.build(model);

  std::vector<unsigned> order;
  for (auto & payload : ensemble_instances) {
    unsigned id = payload.first;
    if (corpus.training_data.count(id) == 0) { continue; }
    order.push_back(id);
  }

  float llh = 0.f;
  float best_acc = -1.f;
  unsigned n_processed = 0;

  _INFO << "[postag|ensemble|train] will stop after " << max_iter << " iterations.";
  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    llh = 0.f;
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits & units = corpus.training_data.at(sid).input_units;
      const EnsembleInstance & inst = ensemble_instances.at(sid);

      {
        dynet::ComputationGraph cg;
        engine.new_graph(cg);

        unsigned n_words = units.size() - 1;
        std::vector<std::string> words(n_words);
        for (unsigned i = 1; i < units.size(); ++i) {
          words[i - 1] = units[i].word;
        }
        engine.initialize(words);
        unsigned prev_label = AlphabetCollection::get()->pos_map.get(Corpus::ROOT);

        std::vector<dynet::Expression> loss;
        const std::vector<unsigned> & actions = inst.categories;
        const std::vector<std::vector<float>> & probs = inst.probs;

        unsigned n_pos = probs.at(0).size();
        for (unsigned i = 0; i < n_words; ++i) {
          dynet::Expression feature = engine.get_feature(i, prev_label);
          dynet::Expression logits = engine.get_emit_score(feature);
          const std::vector<float> & prob = probs.at(i);

          loss.push_back(dynet::dot_product(
            dynet::input(cg, { n_pos }, prob),
            dynet::log_softmax(logits)
          ));
          prev_label = actions.at(i);
        }
        if (!loss.empty()) {
          dynet::Expression loss_expr = -dynet::sum(loss);
          if (lambda_ > 0) { loss_expr = loss_expr + lambda_ * engine.l2(); }
          float l = dynet::as_scalar(cg.forward(loss_expr));
          cg.backward(loss_expr);
          trainer->update();
          llh += l;
          n_processed++;
        }
      }

      if (need_evaluate(iter, n_processed)) {
        float acc = evaluate(corpus);
        float prop = static_cast<float>(n_processed) / order.size();
        if (acc > best_acc) {
          _INFO << "[postag|ensemble|train] " << prop << "% trained, ACC on heldout = " << acc
            << ", new best achieved, saved.";
          best_acc = acc;
          Model::get()->to_json(Model::kPostaggerName, engine.model);
        } else {
          _INFO << "[postag|ensemble|train] " << prop << "% trained, ACC on heldout = " << acc;
        }
      }
    }
    _INFO << "[postag|ensemble|train] end of iter #" << iter << " loss " << llh;
    if (need_evaluate(iter)) {
      float acc = evaluate(corpus);
      if (acc > best_acc) {
        best_acc = acc;
        _INFO << "[postag|train] end of iter #" << iter << ", ACC on heldout = " << acc <<
          ", new best achieved, saved.";
        Model::get()->to_json(Model::kPostaggerName, engine.model);
      } else {
        _INFO << "[postag|train] end of iter #" << iter << ", ACC on heldout = " << acc;
      }
    }
    opt_builder.update(trainer, iter);
  }
}

}
