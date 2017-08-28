#include "parse_model_builder.h"
// #include "parser_dyer15.h"
#include "parse_model_ballesteros15.h"
// #include "parser_kiperwasser16.h"
#include "archybrid.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {


ParseModelBuilder::ParseModelBuilder(po::variables_map & conf) {
  system_name = conf["parse-system"].as<std::string>();
  arch_name = conf["parse-arch"].as<std::string>();

  char_dim = conf["parse-char-dim"].as<unsigned>();
  word_dim = conf["parse-word-dim"].as<unsigned>();
  pos_dim = conf["parse-pos-dim"].as<unsigned>();
  pretrained_dim = conf["parse-pretrained-dim"].as<unsigned>();
  action_dim = conf["parse-action-dim"].as<unsigned>();
  label_dim = conf["parse-label-dim"].as<unsigned>();
  n_layers = conf["parse-n-layer"].as<unsigned>();
  lstm_input_dim = conf["parse-lstm-input-dim"].as<unsigned>();
  hidden_dim = conf["parse-hidden-dim"].as<unsigned>();
}

ParseModel * ParseModelBuilder::build(dynet::ParameterCollection & model) {
  if (system_name == "arcstd") {
    // sys = new ArcStandard(corpus.deprel_map);
  } else if (system_name == "arceager") {
    // sys = new ArcEager(corpus.deprel_map);
  } else if (system_name == "archybrid") {
    system = new ArcHybrid();
  } else if (system_name == "swap") {
    // sys = new Swap(corpus.deprel_map);
  } else {
    _ERROR << "[twpipe|ParseModelBuilder] unknown transition system: " << system_name;
    exit(1);
  }
  _INFO << "[twpipe|ParseModelBuilder] transition system: " << system_name;

  ParseModel* parser = nullptr;

  if (arch_name == "dyer15" || arch_name == "d15") {
    /*parser = new ParserDyer15(model,
                              corpus.training_vocab.size() + 10,
                              conf["word_dim"].as<unsigned>(),
                              corpus.pos_map.size() + 10,
                              conf["pos_dim"].as<unsigned>(),
                              corpus.norm_map.size() + 1,
                              conf["pretrained_dim"].as<unsigned>(),
                              sys.num_actions(),
                              conf["action_dim"].as<unsigned>(),
                              conf["label_dim"].as<unsigned>(),
                              conf["layers"].as<unsigned>(),
                              conf["lstm_input_dim"].as<unsigned>(),
                              conf["hidden_dim"].as<unsigned>(),
                              system_name,
                              sys,
                              pretrained);*/
  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    parser = new Ballesteros15Model(model,
                                    AlphabetCollection::get()->char_map.size() + 10,
                                    char_dim,
                                    word_dim,
                                    AlphabetCollection::get()->pos_map.size() + 10,
                                    pos_dim,
                                    pretrained_dim,
                                    system->num_actions(),
                                    action_dim,
                                    label_dim,
                                    n_layers,
                                    lstm_input_dim,
                                    hidden_dim,
                                    (*system));

  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    /*parser = new ParserKiperwasser16(model,
                                     corpus.training_vocab.size() + 10,
                                     conf["word_dim"].as<unsigned>(),
                                     corpus.pos_map.size() + 10,
                                     conf["pos_dim"].as<unsigned>(),
                                     corpus.norm_map.size() + 1,
                                     conf["pretrained_dim"].as<unsigned>(),
                                     sys.num_actions(),
                                     conf["layers"].as<unsigned>(),
                                     conf["lstm_input_dim"].as<unsigned>(),
                                     conf["hidden_dim"].as<unsigned>(),
                                     system_name,
                                     sys,
                                     pretrained);*/
  } else {
    _ERROR << "Main:: Unknown architecture name: " << arch_name;
    exit(1);
  }
  _INFO << "Main:: architecture: " << arch_name;
  return parser;
}

void ParseModelBuilder::to_json() {
}

ParseModel * ParseModelBuilder::from_json(dynet::ParameterCollection & model) {
  return nullptr;
}

}