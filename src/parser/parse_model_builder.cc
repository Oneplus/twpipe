#include "parse_model_builder.h"
#include "parse_model_dyer15.h"
#include "parse_model_ballesteros15.h"
#include "parse_model_kiperwasser16.h"
#include "archybrid.h"
#include "arcstd.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/model.h"

namespace twpipe {


ParseModelBuilder::ParseModelBuilder(po::variables_map & conf) {
  system_name = conf["parse-system"].as<std::string>();
  arch_name = conf["parse-arch"].as<std::string>();

  char_size = AlphabetCollection::get()->char_map.size();
  word_size = AlphabetCollection::get()->word_map.size();
  pos_size = AlphabetCollection::get()->pos_map.size();
  
  char_dim = conf["parse-char-dim"].as<unsigned>();
  word_dim = conf["parse-word-dim"].as<unsigned>();
  pos_dim = conf["parse-pos-dim"].as<unsigned>();
  action_dim = conf["parse-action-dim"].as<unsigned>();
  label_dim = conf["parse-label-dim"].as<unsigned>();
  n_layers = conf["parse-n-layer"].as<unsigned>();
  lstm_input_dim = conf["parse-lstm-input-dim"].as<unsigned>();
  hidden_dim = conf["parse-hidden-dim"].as<unsigned>();
  embed_dim = conf["embedding-dim"].as<unsigned>();
}

ParseModel * ParseModelBuilder::build(dynet::ParameterCollection & model) {
  if (system_name == "arcstd") {
    system = new ArcStandard();
  } else if (system_name == "arceager") {
    // sys = new ArcEager(corpus.deprel_map);
  } else if (system_name == "archybrid") {
    system = new ArcHybrid();
  } else if (system_name == "swap") {
    // sys = new Swap(corpus.deprel_map);
  } else {
    _ERROR << "[parse|model_builder] unknown transition system: " << system_name;
    exit(1);
  }
  _INFO << "[parse|model_builder] transition system: " << system_name;

  ParseModel* parser = nullptr;

  if (arch_name == "dyer15" || arch_name == "d15") {
    parser = new Dyer15Model(model,
                             word_size,
                             word_dim,
                             pos_size,
                             pos_dim,
                             embed_dim,
                             system->num_actions(),
                             action_dim,
                             label_dim,
                             n_layers,
                             lstm_input_dim,
                             hidden_dim,
                             (*system));

  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    parser = new Ballesteros15Model(model,
                                    char_size,
                                    char_dim,
                                    word_dim,
                                    pos_size,
                                    pos_dim,
                                    embed_dim,
                                    system->num_actions(),
                                    action_dim,
                                    label_dim,
                                    n_layers,
                                    lstm_input_dim,
                                    hidden_dim,
                                    (*system));

  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    parser = new Kiperwasser16Model(model,
                                    word_size,
                                    word_dim,
                                    pos_size,
                                    pos_dim,
                                    embed_dim,
                                    system->num_actions(),
                                    n_layers,
                                    lstm_input_dim,
                                    hidden_dim,
                                    (*system));
  } else {
    _ERROR << "[parse|model_builder] unknown architecture name: " << arch_name;
    exit(1);
  }
  _INFO << "[parse|model_builder] architecture: " << arch_name;
  return parser;
}

void ParseModelBuilder::to_json() {
  Model::get()->to_json(Model::kParserName, {
    { "system", system_name },
    { "arch", arch_name },
    { "pos-dim", boost::lexical_cast<std::string>(pos_dim) },
    { "n-postags", boost::lexical_cast<std::string>(pos_size) },
    { "lstm-input-dim", boost::lexical_cast<std::string>(lstm_input_dim) },
    { "hidden-dim", boost::lexical_cast<std::string>(hidden_dim) },
    { "emb-dim", boost::lexical_cast<std::string>(embed_dim) }
  });

  if (arch_name == "dyer15" || arch_name == "d15") {
    Model::get()->to_json(Model::kParserName, {
      { "n-words", boost::lexical_cast<std::string>(word_size) },
      { "word-dim", boost::lexical_cast<std::string>(word_dim) },
      { "action-dim", boost::lexical_cast<std::string>(action_dim)},
      { "label-dim", boost::lexical_cast<std::string>(label_dim) }
    });

  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    Model::get()->to_json(Model::kParserName, {
      { "n-chars", boost::lexical_cast<std::string>(char_size) },
      { "char-dim", boost::lexical_cast<std::string>(char_dim) },
      { "action-dim", boost::lexical_cast<std::string>(action_dim)},
      { "label-dim", boost::lexical_cast<std::string>(label_dim) }
    });

  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    Model::get()->to_json(Model::kParserName, {
      { "n-words", boost::lexical_cast<std::string>(word_size) },
      { "word-dim", boost::lexical_cast<std::string>(word_dim) },
    });

  } else {
    _ERROR << "[parse|model_builder] unknown architecture name: " << arch_name;
    exit(1);
  }
}

ParseModel * ParseModelBuilder::from_json(dynet::ParameterCollection & model) {
  ParseModel * engine = nullptr;

  Model * globals = Model::get();
  system_name = globals->from_json(Model::kParserName, "system");

  BOOST_ASSERT_MSG(system == nullptr,
                   "[parse|model_builder] transition system should not be initialized.");
  if (system_name == "arcstd") {
    system = new ArcStandard();
  } else if (system_name == "arceager") {
    // sys = new ArcEager(corpus.deprel_map);
  } else if (system_name == "archybrid") {
    system = new ArcHybrid();
  } else if (system_name == "swap") {
    // sys = new Swap(corpus.deprel_map);
  } else {
    _ERROR << "[parse|model_builder] unknown transition system: " << system_name;
    exit(1);
  }
  _INFO << "[parse|model_builder] transition system: " << system_name;
  arch_name = globals->from_json(Model::kParserName, "arch");

  unsigned temp_size;
  temp_size =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "n-postags"));
  if (pos_size == 0) {
    pos_size = temp_size;
  } else {
    BOOST_ASSERT_MSG(pos_size == temp_size, "[parser|model_builder] pos-size mismatch!");
  }

  pos_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "pos-dim"));
  lstm_input_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "lstm-input-dim"));
  hidden_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "hidden-dim"));
  embed_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "emb-dim"));

  if (arch_name == "dyer15" || arch_name == "d15") {
    temp_size =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "n-words"));
    if (word_size == 0) {
      word_size = temp_size;
    } else {
      BOOST_ASSERT_MSG(word_size == temp_size, "[parse|model_builder] word-size mismatch!");
    }
    
    word_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "word-dim"));
    action_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "action-dim"));
    label_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "label-dim"));

  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    char_size =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "n-chars"));
    if (char_size == 0) {
      char_size = char_size;
    } else {
      BOOST_ASSERT_MSG(char_size == temp_size, "[parse|model_builder] word-size mismatch!");
    }
    
    char_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "char-dim"));
    action_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "action-dim"));
    label_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "label-dim"));

  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    temp_size =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "n-words"));
    if (word_size == 0) {
      word_size = temp_size;
    } else {
      BOOST_ASSERT_MSG(word_size == temp_size, "[parse|model_builder] word-size mismatch!");
    }

    word_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kParserName, "word-dim"));
    
  } else {
    _ERROR << "[parse|model_builder] unknown architecture name: " << arch_name;
    exit(1);
  }
  engine = build(model);
  globals->from_json(Model::kParserName, model);
  return engine;
}

}