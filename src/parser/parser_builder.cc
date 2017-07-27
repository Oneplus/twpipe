#include "parser_dyer15.h"
#include "parser_ballesteros15.h"
#include "parser_kiperwasser16.h"
#include "parser_builder.h"
#include "logging.h"

po::options_description ParserBuilder::get_options() {
  po::options_description cmd("Parser settings.");
  return cmd;
}

Parser * ParserBuilder::build(const po::variables_map& conf,
                              dynet::Model& model,
                              TransitionSystem& sys,
                              const Corpus& corpus,
                              const std::unordered_map<unsigned, std::vector<float>>& pretrained) {
  std::string system_name = conf["system"].as<std::string>();
  Parser* parser = nullptr;
  std::string arch_name = conf["architecture"].as<std::string>();
  if (arch_name == "dyer15" || arch_name == "d15") {
    parser = new ParserDyer15(model,
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
                              pretrained);
  } else if (arch_name == "ballesteros15" || arch_name == "b15") {
    parser = new ParserBallesteros15(model,
                                     corpus.char_map.size() + 10,
                                     conf["char_dim"].as<unsigned>(),
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
                                     pretrained);

  } else if (arch_name == "kiperwasser16" || arch_name == "k16") {
    parser = new ParserKiperwasser16(model,
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
                                     pretrained);
  } else {
    _ERROR << "Main:: Unknown architecture name: " << arch_name;
    exit(1);
  }
  _INFO << "Main:: architecture: " << arch_name;
  return parser;
}
