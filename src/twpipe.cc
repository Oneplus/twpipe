#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include "tokenizer/tokenize_model.h"
#include "tokenizer/tokenize_model_builder.h"
#include "tokenizer/tokenizer_trainer.h"
#include "postagger/postag_model.h"
#include "postagger/postag_model_builder.h"
#include "postagger/postagger_trainer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet.h"
#include "twpipe/corpus.h"
#include "twpipe/optimizer_builder.h"
#include "twpipe/trainer.h"
#include "twpipe/model.h"

namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description generic_opts("Generic options");
  generic_opts.add_options()
    ("verbose,v", "Details logging.")
    ("help,h", "show help information.")
    ("train", "use to specify training.")
    ("input-file", po::value<std::string>(), "input files")
    ;

  po::options_description running_opts("Running options");
  running_opts.add_options()
    ("tokenize", "perform tokenization")
    ("tag", "perform tagging")
    ("parse", "perform parsing")
    ("format", po::value<std::string>()->default_value("plain"), "the format of input data [plain|text].")
    ;

  po::options_description model_opts = twpipe::Model::get_options();
  po::options_description training_opts = twpipe::Trainer::get_options();
  po::options_description tokenizer_opts = twpipe::TokenizeModel::get_options();
  po::options_description postagger_opts = twpipe::PostagModel::get_options();
  po::options_description optimizer_opts = twpipe::OptimizerBuilder::get_options();

  po::positional_options_description input_opts;
  input_opts.add("input-file", -1);

  po::options_description cmd("Usage: ./twpipe [running_opts] model_file [input_file]\n"
                              "       ./twpipe --train [training_opts] model_file [input_file]");
  cmd.add(generic_opts)
    .add(running_opts)
    .add(model_opts)
    .add(training_opts)
    .add(tokenizer_opts)
    .add(postagger_opts)
    .add(optimizer_opts)
    ;

  po::store(po::command_line_parser(argc, argv).options(cmd).positional(input_opts).run(),
            conf);
  po::notify(conf);

  if (conf.count("help")) {
    std::cerr << cmd << std::endl;
    exit(1);
  }
  twpipe::init_boost_log(conf.count("verbose") > 0);
  
  if (!conf.count("input-file")) {
    std::cerr << "Please specify input file." << std::endl;
    exit(1);
  }
}

int main(int argc, char* argv[]) {
  dynet::initialize(argc, argv);

  po::variables_map conf;
  init_command_line(argc, argv, conf);

  if (conf.count("train")) {
    twpipe::Corpus corpus;
    corpus.load_training_data(conf["input-file"].as<std::string>());

    twpipe::Model::get()->to_json("char-map", corpus.char_map);
    twpipe::Model::get()->to_json("pos-map", corpus.pos_map);
    twpipe::Model::get()->to_json("deprel-map", corpus.deprel_map);

    if (conf.count("heldout")) {
      corpus.load_devel_data(conf["heldout"].as<std::string>());
    }

    twpipe::OptimizerBuilder opt_builder(conf);

    if (conf["train-tokenizer"].as<bool>() == true) {
      _INFO << "[twpipe] going to train tokenizer.";
      
      dynet::ParameterCollection model;

      twpipe::TokenizeModelBuilder builder(conf, corpus.char_map);
      builder.to_json();

      twpipe::TokenizeModel * engine = builder.build(model);
      twpipe::TokenizerTrainer trainer(*engine, opt_builder, conf);
      trainer.train(corpus);
    }
    if (conf["train-tagger"].as<bool>() == true) {
      _INFO << "[twpipe] going to train postagger.";

      dynet::ParameterCollection model;

      twpipe::PostagModelBuilder builder(conf, corpus.char_map, corpus.pos_map);
      builder.to_json();

      twpipe::PostagModel * engine = builder.build(model);
      twpipe::PostaggerTrainer trainer(*engine, opt_builder,
              twpipe::StrEmbeddingType(), conf);
      trainer.train(corpus);
    }
    if (conf["train-parser"].as<bool>() == true) {

    }

    std::string model_name = conf["model"].as<std::string>();
    twpipe::Model::get()->save(model_name);
  } else {
    std::string model_name = conf["model"].as<std::string>();
    twpipe::Model::get()->load(model_name);

    twpipe::Alphabet char_map;
    twpipe::Alphabet pos_map;
    twpipe::Alphabet deprel_map;

    twpipe::Model::get()->from_json("char-map", char_map);
    twpipe::Model::get()->from_json("pos-map", pos_map);
    twpipe::Model::get()->from_json("deprel-map", deprel_map);

    if (conf.count("tokenize")) {
      if (!twpipe::Model::get()->has_tokenizer_model()) {
        _ERROR << "[twpipe] doesn't have tokenizer model!";
        exit(1);
      }
      twpipe::TokenizeModelBuilder builder(conf, char_map);
      dynet::ParameterCollection model;
      twpipe::TokenizeModel * engine = builder.from_json(model);

      std::ifstream ifs(conf["input-file"].as<std::string>());
      std::string buffer;
      while (std::getline(ifs, buffer)) {
        boost::algorithm::trim(buffer);
        engine->tokenize(buffer);
      }
    }
  }
  return 0;
}
