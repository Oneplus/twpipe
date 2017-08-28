#include "parser_trainer.h"
#include "tree.h"
// #include "evaluate.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

po::options_description SupervisedTrainer::get_options() {
  po::options_description cmd("Parser supervised learning options");
  cmd.add_options()
    ("parse-supervised-oracle", po::value<std::string>()->default_value("static"), "The type of oracle in supervised learning [static|dynamic].")
    ("parse-supervised-objective", po::value<std::string>()->default_value("crossentropy"), "The learning objective [crossentropy|rank|bipartie_rank|structure]")
    ("parse-supervised-do-pretrain-iter", po::value<unsigned>()->default_value(1), "The number of pretrain iteration on dynamic oracle.")
    ("parse-supervised-do-explore-prob", po::value<float>()->default_value(0.9), "The probability of exploration.")
    ("parse-noisify-method", po::value<std::string>()->default_value("none"), "The type of noisifying method [none|singleton|word]")
    ("parse-noisify-singleton-dropout-prob", po::value<float>()->default_value(0.2f), "The probability of dropping singleton, used in singleton mode.")
    ;
  return cmd;
}

SupervisedTrainer::SupervisedTrainer(ParseModel & engine,
                                     OptimizerBuilder & opt_builder,
                                     const po::variables_map& conf) :
  Trainer(conf),
  engine(engine),
  opt_builder(opt_builder) {

  std::string supervised_oracle = conf["parse-supervised-oracle"].as<std::string>();
  if (supervised_oracle == "static") {
    oracle_type = kStatic;
  } else if (supervised_oracle == "dynamic") {
    oracle_type = kDynamic;
  } else {
    _ERROR << "[twpipe|parser] unknown oracle :" << supervised_oracle;
  }

  std::string supervised_objective_name = conf["parse-supervised-objective"].as<std::string>();
  if (supervised_objective_name == "crossentropy") {
    objective_type = kCrossEntropy;
  } else if (supervised_objective_name == "rank") {
    objective_type = kRank;
  } else if (supervised_objective_name == "bipartie_rank") {
    objective_type = kBipartieRank;
  } else {
    objective_type = kStructure;
    if (!conf.count("parse-beam-size") || conf["parse-beam-size"].as<unsigned>() <= 1) {
      _ERROR << "[twpipe|parser] set structure learning objective, but parse-beam-size was not set.";
      exit(1);
    }
  }
  _INFO << "[twpipe|parser] learning objective " << supervised_objective_name;

  if (oracle_type == kDynamic) {
    do_pretrain_iter = conf["parse-supervised-do-pretrain-iter"].as<unsigned>();
    do_explore_prob = conf["parse-supervised-do-explore-prob"].as<float>();
    _INFO << "[twpipe|parser] use dynamic oracle training";
    _INFO << "[twpipe|parser] pretrain iteration = " << do_pretrain_iter;
    _INFO << "[twpipe|parser] explore prob = " << do_explore_prob;
  }

  beam_size = (conf.count("parse-beam-size") ? conf["parse-beam-size"].as<unsigned>() : 0);
  allow_nonprojective = (conf["parse-system"].as<std::string>() == "swap");
  
  /// save noisifier configuration.
  noisify_method_name = conf["parse-noisify-method"].as<std::string>();
  singleton_dropout_prob = conf["parse-noisify-singleton-dropout-prob"].as<float>();
}

void SupervisedTrainer::train(Corpus& corpus) {
  _INFO << "[twpipe|parser] start lstm-parser supervised training.";
  Noisifier noisifier(corpus, noisify_method_name, singleton_dropout_prob);
  
  dynet::Trainer* trainer = opt_builder.build(engine.model);
  float eta0 = trainer->learning_rate;
  unsigned kUNK = AlphabetCollection::get()->word_map.get(Corpus::UNK);

  float llh = 0.f;
  float llh_in_batch = 0.f;
  float best_las = -1.f;

  std::vector<unsigned> order;
  get_orders(corpus, order, allow_nonprojective);
  float n_train = order.size();

  unsigned logc = 0;
  bool use_beam_search = (beam_size > 1);
  _INFO << "[twpipe|parser] will stop after " << max_iter << " iterations.";
  
  for (unsigned iter = 1; iter <= max_iter; ++iter) {
    llh = 0;
    _INFO << "[twpipe|parser] start training iteration #" << iter << ", shuffled.";
    std::shuffle(order.begin(), order.end(), (*dynet::rndeng));

    for (unsigned sid : order) {
      InputUnits& input_units = corpus.training_data[sid].input_units;
      const ParseUnits& parse_units = corpus.training_data[sid].parse_units;

      noisifier.noisify(input_units);
      float lp;
      if (objective_type == kStructure) {
        lp = train_structure_full_tree(input_units, parse_units, trainer, beam_size);
      } else {
        lp = train_full_tree(input_units, parse_units, trainer, iter);
      }
      llh += lp;
      llh_in_batch += lp;
      noisifier.denoisify(input_units);
    }

    _INFO << "[twpipe|parser] end of iter #" << iter << " loss " << llh;
    float n_recall = 0., n_total = 0.;
    for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
      const Instance & inst = corpus.devel_data.at(sid);

      unsigned len = inst.input_units.size();
      std::vector<std::string> words(len - 1), postags(len - 1);
      std::vector<unsigned> pred_heads, gold_heads(len - 1);
      std::vector<std::string> pred_deprels, gold_deprels(len - 1);
      for (unsigned i = 1; i < inst.input_units.size(); ++i) {
        words[i - 1] = inst.input_units[i].word;
        postags[i - 1] = inst.input_units[i].postag;
        gold_heads[i - 1] = inst.parse_units[i].head;
        gold_deprels[i - 1] = AlphabetCollection::get()->deprel_map.get(
          inst.parse_units[i].deprel);
      }
      engine.predict(words, postags, pred_heads, pred_deprels);
      for (unsigned i = 0; i < pred_heads.size(); ++i) {
        if (gold_heads[i] == pred_heads[i] &&
            gold_deprels[i] == pred_deprels[i]) { n_recall += 1.; }
        n_total += 1.;
      }
    }
    float las = n_recall / n_total;
    if (las > best_las) {
      best_las = las;
      _INFO << "[twpipe|parser] new best record achieved: " << best_las << ", saved.";
      // dynet::save_dynet_model(name, (&model));
    }
    trainer->learning_rate = eta0 / (1. + static_cast<float>(iter) * 0.08);
  }

  delete trainer;
}

void SupervisedTrainer::add_loss_one_step(dynet::Expression & score_expr,
                                          const unsigned & best_gold_action,
                                          const unsigned & worst_gold_action,
                                          const unsigned & best_non_gold_action,
                                          std::vector<dynet::Expression> & loss) {
  TransitionSystem & sys = engine.sys;
  unsigned illegal_action = sys.num_actions();

  if (objective_type == kCrossEntropy) {
    loss.push_back(dynet::pickneglogsoftmax(score_expr, best_gold_action));
  } else if (objective_type == kRank) {
    if (best_gold_action != illegal_action && best_non_gold_action != illegal_action) {
      loss.push_back(dynet::pairwise_rank_loss(
        dynet::pick(score_expr, best_gold_action),
        dynet::pick(score_expr, best_non_gold_action)
      ));
    }
  } else {
    if (worst_gold_action != illegal_action && best_non_gold_action != illegal_action) {
      loss.push_back(dynet::pairwise_rank_loss(
        dynet::pick(score_expr, worst_gold_action),
        dynet::pick(score_expr, best_non_gold_action)
      ));
    }
  }
}

float SupervisedTrainer::train_full_tree(const InputUnits& input_units,
                                         const ParseUnits& parse_units,
                                         dynet::Trainer* trainer,
                                         unsigned iter) {
  TransitionSystem & sys = engine.sys;

  std::vector<unsigned> ref_heads, ref_deprels;
  parse_to_vector(parse_units, ref_heads, ref_deprels);

  dynet::ComputationGraph cg;
  engine.new_graph(cg);
  std::vector<dynet::Expression> loss;
  std::vector<unsigned> gold_actions;
  sys.get_oracle_actions(ref_heads, ref_deprels, gold_actions);

  unsigned len = input_units.size();
  State state(len);
  ParseModel::StateCheckpoint * checkpoint = engine.get_initial_checkpoint();
  engine.initialize(cg, input_units, state, checkpoint);
  unsigned illegal_action = sys.num_actions();
  unsigned n_actions = 0;
  while (!state.terminated()) {
    // collect all valid actions.
    std::vector<unsigned> valid_actions;
    sys.get_valid_actions(state, valid_actions);

    dynet::Expression score_exprs = engine.get_scores(checkpoint);
    std::vector<float> scores = dynet::as_vector(cg.get_value(score_exprs));
    unsigned action = 0;

    unsigned best_gold_action = illegal_action;
    unsigned worst_gold_action = illegal_action;
    unsigned best_non_gold_action = illegal_action;

    if (oracle_type == kDynamic) {
      auto payload = ParseModel::get_best_action(scores, valid_actions);
      action = payload.first;
      std::vector<float> costs; // the larger, the better
      sys.get_transition_costs(state, valid_actions, ref_heads, ref_deprels, costs);
      float gold_action_cost = (*std::max_element(costs.begin(), costs.end()));
      float action_cost = 0.f;
      float best_gold_action_score = -1e10, worst_gold_action_score = 1e10, best_non_gold_action_score = -1e10;
      for (unsigned i = 0; i < valid_actions.size(); ++i) {
        unsigned act = valid_actions[i];
        float s = scores[act];
        if (costs[i] == gold_action_cost) {
          if (best_gold_action_score < s) { best_gold_action_score = s; best_gold_action = act; }
          if (worst_gold_action_score > s) { worst_gold_action_score = s; worst_gold_action = act; }
        } else {
          if (best_non_gold_action_score < s) { best_non_gold_action_score = s; best_non_gold_action = act; }
        }
        if (act == action) { action_cost = costs[i]; }
      }
      if (gold_action_cost != action_cost) {
        if (!(iter >= do_pretrain_iter && dynet::rand01() < do_explore_prob)) {
          action = best_gold_action;
        }
      }
    } else {
      best_gold_action = gold_actions[n_actions];
      action = gold_actions[n_actions];
      if (objective_type == kRank || objective_type == kBipartieRank) {
        float best_non_gold_action_score = -1e10;
        for (unsigned i = 0; i < valid_actions.size(); ++i) {
          unsigned act = valid_actions[i];
          if (act != best_gold_action && (scores[act] > best_non_gold_action_score)) {
            best_non_gold_action = act;
            best_non_gold_action_score = scores[act];
          }
        }
      }
    }

    add_loss_one_step(score_exprs,
                      best_gold_action,
                      worst_gold_action,
                      best_non_gold_action,
                      loss);
    engine.perform_action(action, cg, state, checkpoint);
    n_actions++;
  }
  engine.destropy_checkpoint(checkpoint);
  float ret = 0.;
  if (loss.size() > 0) {
    dynet::Expression l = dynet::sum(loss);
    ret = dynet::as_scalar(cg.forward(l));
    cg.backward(l);
    trainer->update();
  }
  return ret;
}

float SupervisedTrainer::train_structure_full_tree(const InputUnits & input_units,
                                                   const ParseUnits & parse_units,
                                                   dynet::Trainer * trainer,
                                                   unsigned beam_size) {
  typedef std::tuple<unsigned, unsigned, float, dynet::Expression> Transition;
  TransitionSystem & sys = engine.sys;

  dynet::ComputationGraph cg;
  engine.new_graph(cg);

  std::vector<unsigned> gold_heads, gold_deprels, gold_actions;
  parse_to_vector(parse_units, gold_heads, gold_deprels);
  sys.get_oracle_actions(gold_heads, gold_deprels, gold_actions);

  unsigned len = input_units.size();
  std::vector<State> states;
  std::vector<float> scores;
  std::vector<dynet::Expression> scores_exprs;
  std::vector<ParseModel::StateCheckpoint *> checkpoints;

  states.push_back(State(len));
  scores.push_back(0.);
  scores_exprs.push_back(dynet::zeroes(cg, { 1 }));
  checkpoints.push_back(engine.get_initial_checkpoint());
  engine.initialize(cg, input_units, states[0], checkpoints[0]);

  unsigned curr = 0, next = 1, corr = 0;
  unsigned n_step = 0;
  while (!states[corr].terminated()) {
    unsigned gold_action = gold_actions[n_step];
    n_step++;

    std::vector<Transition> transitions;
    for (unsigned i = curr; i < next; ++i) {
      const State& prev_state = states[i];
      float prev_score = scores[i];
      dynet::Expression prev_score_expr = scores_exprs[i];

      if (prev_state.terminated()) {
        transitions.push_back(std::make_tuple(
          i, sys.num_actions(), prev_score, prev_score_expr
        ));
      } else {
        ParseModel::StateCheckpoint * checkpoint = checkpoints[i];
        std::vector<unsigned> valid_actions;
        sys.get_valid_actions(prev_state, valid_actions);

        dynet::Expression transit_scores_expr = engine.get_scores(checkpoint);
        std::vector<float> transit_scores = dynet::as_vector(cg.get_value(transit_scores_expr));
        for (unsigned a : valid_actions) {
          transitions.push_back(std::make_tuple(
            i, a, prev_score + transit_scores[a],
            prev_score_expr + dynet::pick(transit_scores_expr, a)
          ));
        }
      }
    }

    sort(transitions.begin(), transitions.end(),
         [](const Transition& a, const Transition& b) { return std::get<2>(a) > std::get<2>(b); });

    unsigned new_corr = UINT_MAX, new_curr = next, new_next = next;
    for (unsigned i = 0; i < transitions.size() && i < beam_size; ++i) {
      unsigned cursor = std::get<0>(transitions[i]);
      unsigned action = std::get<1>(transitions[i]);
      float new_score = std::get<2>(transitions[i]);
      dynet::Expression new_score_expr = std::get<3>(transitions[i]);
      State& state = states[cursor];

      State new_state(state);
      ParseModel::StateCheckpoint * new_checkpoint = engine.copy_checkpoint(checkpoints[cursor]);
      if (action != sys.num_actions()) {
        engine.perform_action(action, cg, new_state, new_checkpoint);
      }

      //      
      states.push_back(new_state);
      scores.push_back(new_score);
      scores_exprs.push_back(new_score_expr);
      checkpoints.push_back(new_checkpoint);

      if (cursor == corr && action == gold_action) { new_corr = new_next; }
      new_next++;
    }
    if (new_corr == UINT_MAX) {
      // early stopping
      break;
    } else {
      corr = new_corr;
      curr = new_curr;
      next = new_next;
    }
  }

  for (ParseModel::StateCheckpoint * checkpoint : checkpoints) {
    engine.destropy_checkpoint(checkpoint);
  }

  std::vector<dynet::Expression> loss;
  for (unsigned i = curr; i < next; ++i) {
    loss.push_back(scores_exprs[i]);
  }
  dynet::Expression l = dynet::pickneglogsoftmax(dynet::concatenate(loss), corr - curr);
  float ret = dynet::as_scalar(cg.forward(l));
  cg.backward(l);
  trainer->update();
  return ret;
}

void SupervisedTrainer::get_orders(Corpus& corpus,
                                   std::vector<unsigned>& order,
                                   bool non_projective) {
  order.clear();
  for (unsigned i = 0; i < corpus.training_data.size(); ++i) {
    const ParseUnits& parse_units = corpus.training_data[i].parse_units;
    if (!DependencyUtils::is_tree(parse_units)) {
      _INFO << "[twpipe|parser|get_orders] #" << i << " not a tree, skipped.";
      continue;
    }
    if (!non_projective && !DependencyUtils::is_projective(parse_units)) {
      _INFO << "[twpipe|parser|get_orders] #" << i << " not projective, skipped.";
      continue;
    }
    order.push_back(i);
  }
}

}