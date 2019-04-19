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
#include "parser/parse_model.h"
#include "parser/parse_model_builder.h"
#include "parser/parser_trainer.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"
#include "twpipe/corpus.h"
#include "twpipe/optimizer_builder.h"
#include "twpipe/trainer.h"
#include "twpipe/model.h"
#include "twpipe/elmo.h"
#include "twpipe/embedding.h"
#include "twpipe/cluster.h"

namespace po = boost::program_options;

void init_command_line(int argc, char* argv[], po::variables_map& conf) {
  po::options_description generic_opts("Generic options");
  generic_opts.add_options()
    ("verbose,v", "Details logging.")
    ("help,h", "show help information.")
    ("train", "use to specify training.")
    ("input-file", po::value<std::string>(), "the path to the input file.")
    ;

  po::options_description running_opts("Running options");
  running_opts.add_options()
    ("segment-and-tokenize", "perform sentence split and tokenization.")
    ("tokenize", "perform tokenization")
    ("postag", "perform tagging")
    ("parse", "perform parsing")
    ("format", po::value<std::string>()->default_value("plain"), "the format of input data [plain|conll].")
    ;

  po::options_description model_opts = twpipe::Model::get_options();
  po::options_description embed_opts = twpipe::WordEmbedding::get_options();
  po::options_description elmo_opts = twpipe::ELMo::get_options();
  po::options_description training_opts = twpipe::Trainer::get_options();
  po::options_description tokenizer_opts = twpipe::AbstractTokenizeModel::get_options();
  po::options_description postagger_opts = twpipe::PostagModel::get_options();
  po::options_description postagger_ensemble_train_opts = twpipe::PostaggerEnsembleTrainer::get_options();
  po::options_description parser_opts = twpipe::ParseModel::get_options();
  po::options_description parser_train_opts = twpipe::ParserTrainer::get_options();
  po::options_description parser_supervised_train_opts = twpipe::SupervisedTrainer::get_options();
  po::options_description parser_ensemble_train_opts = twpipe::SupervisedEnsembleTrainer::get_options();
  po::options_description optimizer_opts = twpipe::OptimizerBuilder::get_options();

  po::positional_options_description input_opts;
  input_opts.add("input-file", -1);

  po::options_description cmd("Usage: ./twpipe [running_opts] model_file [input_file]\n"
                              "       ./twpipe --train [training_opts] model_file [input_file]");
  cmd.add(generic_opts)
    .add(running_opts)
    .add(model_opts)
    .add(elmo_opts)
    .add(embed_opts)
    .add(training_opts)
    .add(tokenizer_opts)
    .add(postagger_opts)
    .add(postagger_ensemble_train_opts)
    .add(parser_opts)
    .add(parser_supervised_train_opts)
    .add(parser_ensemble_train_opts)
    .add(parser_train_opts)
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

  if (conf.count("embedding")) {
    twpipe::WordEmbedding::get()->load(conf["embedding"].as<std::string>(),
                                       conf["embedding-dim"].as<unsigned>());
  } else {
    twpipe::WordEmbedding::get()->empty(conf["embedding-dim"].as<unsigned>());
  }

  if (conf.count("elmo")) {
    twpipe::ELMo::get()->load(conf["elmo"].as<std::string>(),
                              conf["elmo-dim"].as<unsigned>());
  } else {
    twpipe::ELMo::get()->empty(conf["elmo-dim"].as<unsigned>());
  }

  if (conf.count("train")) {
    twpipe::Corpus corpus;
    corpus.load_training_data(conf["input-file"].as<std::string>());

    if (conf.count("heldout")) {
      corpus.load_devel_data(conf["heldout"].as<std::string>());
    }
    twpipe::AlphabetCollection::get()->to_json();

    twpipe::OptimizerBuilder opt_builder(conf);

    if (conf["train-tokenizer"].as<bool>()) {
      _INFO << "[twpipe] going to train tokenizer.";
      dynet::ParameterCollection model;

      twpipe::TokenizeModelBuilder builder(conf);
      builder.to_json();

      twpipe::TokenizeModel * engine = builder.build(model);
      twpipe::TokenizerTrainer trainer(*engine, opt_builder, conf);
      trainer.train(corpus);
    } else if (conf["train-segmentor-and-tokenizer"].as<bool>()) {
      _INFO << "[twpipe] going to train sentence segmentor and tokenizer.";
      dynet::ParameterCollection model;

      twpipe::SentenceSegmentAndTokenizeModelBuilder builder(conf);
      builder.to_json();

      twpipe::SentenceSegmentAndTokenizeModel * engine = builder.build(model);
      twpipe::TokenizerTrainer trainer(*engine, opt_builder, conf);
      trainer.train(corpus);
    }

    if (conf["train-postagger"].as<bool>()) {
      _INFO << "[twpipe] going to train postagger.";

      dynet::ParameterCollection model;

      twpipe::PostagModelBuilder builder(conf);
      builder.to_json();

      twpipe::PostagModel * engine = builder.build(model);
      if (!conf["train-distill-postagger"].as<bool>()) {
        twpipe::PostaggerTrainer trainer(*engine, opt_builder, conf);
        trainer.train(corpus);
      } else {
        twpipe::EnsembleInstances instances;
        twpipe::EnsembleUtils::load_ensemble_instances(
          conf["pos-ensemble-data"].as<std::string>(),
          instances);
        twpipe::PostaggerEnsembleTrainer trainer((*engine), opt_builder, conf);
        trainer.train(corpus, instances);
      }
    }
    if (conf["train-parser"].as<bool>()) {
      _INFO << "[twpipe] going to train parser.";

      dynet::ParameterCollection model;
      twpipe::ParseModelBuilder builder(conf);
      builder.to_json();
        
      twpipe::ParseModel * engine = builder.build(model);
      if (!conf["train-distill-parser"].as<bool>()) {
        twpipe::SupervisedTrainer trainer((*engine), opt_builder, conf);
        trainer.train(corpus);
      } else {
        twpipe::EnsembleInstances instances;
        twpipe::EnsembleUtils::load_ensemble_instances(
          conf["parse-ensemble-data"].as<std::string>(),
          instances);
        twpipe::SupervisedEnsembleTrainer trainer((*engine), opt_builder, conf);
        trainer.train(corpus, instances);
      }
    }

    std::string model_name = conf["model"].as<std::string>();
    twpipe::Model::get()->save(model_name);
  } else {
    std::string model_name = conf["model"].as<std::string>();
    twpipe::Model::get()->load(model_name);
    twpipe::AlphabetCollection::get()->from_json();

    if (conf["format"].as<std::string>() == "plain") {
      twpipe::TokenizeModel * tok_engine = nullptr;
      twpipe::SentenceSegmentAndTokenizeModel * seg_tok_engine = nullptr;
      twpipe::PostagModel * pos_engine = nullptr;
      twpipe::ParseModel * par_engine = nullptr;
        
      dynet::ParameterCollection tok_model;
      dynet::ParameterCollection seg_tok_model;
      dynet::ParameterCollection pos_model;
      dynet::ParameterCollection par_model;

      bool load_segment_and_tokenize_model = (conf.count("segment-and-tokenize") ||
        conf.count("postag") || conf.count("parse"));
      bool load_tokenize_model = (conf.count("tokenize") > 0);
      bool load_postag_model = (conf.count("postag") || conf.count("parse"));
      bool load_parse_model = (conf.count("parse") > 0);

      if (load_tokenize_model) {
        if (!twpipe::Model::get()->has_tokenizer_model()) {
          _ERROR << "[twpipe] doesn't have tokenizer model!";
          exit(1);
        }
        twpipe::TokenizeModelBuilder tok_builder(conf);
        tok_engine = tok_builder.from_json(tok_model);
      }
      if (load_segment_and_tokenize_model) {
        if (!twpipe::Model::get()->has_segmentor_and_tokenizer_model()) {
          _ERROR << "[twpipe] doesn't have sentence split and tokenizer model!";
          exit(1);
        }
        twpipe::SentenceSegmentAndTokenizeModelBuilder sent_tok_builder(conf);
        seg_tok_engine = sent_tok_builder.from_json(seg_tok_model);
      }
      if (load_postag_model) {
        if (!twpipe::Model::get()->has_postagger_model()) {
          _ERROR << "[twpipe] doesn't have postagger model!";
          exit(1);
        }
        twpipe::PostagModelBuilder pos_builder(conf);
        pos_engine = pos_builder.from_json(pos_model);
      }
      if (load_parse_model) {
        if (!twpipe::Model::get()->has_parser_model()) {
          _ERROR << "[twpipe] doesn't have parser model";
          exit(1);
        }
        twpipe::ParseModelBuilder par_builder(conf);
        par_engine = par_builder.from_json(par_model);
      }

      std::string buffer;
      std::ifstream ifs(conf["input-file"].as<std::string>());
      while (std::getline(ifs, buffer)) {
        boost::algorithm::trim(buffer);
        if (seg_tok_engine != nullptr) {
          std::vector<std::vector<std::string>> sentences;
          seg_tok_engine->sentsegment_and_tokenize(buffer, sentences);

          std::vector<std::string> postags;
          std::vector<unsigned> heads;
          std::vector<std::string> deprels;

          for (unsigned s = 0; s < sentences.size(); ++s) {
            const std::vector<std::string> & tokens = sentences[s];

            if (pos_engine != nullptr) {
              pos_engine->postag(tokens, postags);
            }
            if (par_engine != nullptr) {
              par_engine->predict(tokens, postags, heads, deprels);
            }
            if (s == 0) {
              std::cout << "# text = " << buffer << "\n";
            }
            std::cout << "# sent_id = " << s + 1 << "\n";
            for (unsigned i = 0; i < tokens.size(); ++i) {
              std::cout << i + 1 << "\t" << tokens[i] << "\t_\t"
                        << (pos_engine != nullptr ? postags[i] : "_") << "\t_\t_\t"
                        << (par_engine != nullptr ? std::to_string(heads[i]) : "_") << "\t"
                        << (par_engine != nullptr ? deprels[i] : "_") << "\t_\t_\n";
            }
            std::cout << "\n";
          }
        } else if (tok_engine != nullptr) {
          std::vector<std::string> tokens;
          tok_engine->tokenize(buffer, tokens);

          std::cout << "# text = " << buffer << "\n";
          for (unsigned i = 0; i < tokens.size(); ++i) {
            std::cout << i + 1 << "\t" << tokens[i] << "\t_\t_\t_\t_\t_\t_\t_\t_\n";
          }
          std::cout << "\n";
        }
      }
    } else {
      // for conll format, tokenization is impossible.
      twpipe::PostagModel * pos_engine = nullptr;
      twpipe::ParseModel * par_engine = nullptr;

      dynet::ParameterCollection pos_model;
      dynet::ParameterCollection par_model;

      bool load_postag_model = (conf.count("postag") > 0);
      bool load_parse_model = (conf.count("parse") > 0);
      
      if (load_postag_model) {
        if (!twpipe::Model::get()->has_postagger_model()) {
          _ERROR << "[twpipe] doesn't have postagger model!";
          exit(1);
        }
        twpipe::PostagModelBuilder pos_builder(conf);
        pos_engine = pos_builder.from_json(pos_model);
      }

      if (load_parse_model) {
        if (!twpipe::Model::get()->has_parser_model()) {
          _ERROR << "[twpipe] doesn't have parser model!";
          exit(1);
        }
        twpipe::ParseModelBuilder par_builder(conf);
        par_engine = par_builder.from_json(par_model);
      }
  
      std::vector<std::string> tokens;
      std::vector<std::string> postags, gold_postags;
      std::vector<unsigned> heads, gold_heads;
      std::vector<std::string> deprels, gold_deprels;
      std::string sentence;
      std::string header;
      std::string buffer;
      std::ifstream ifs(conf["input-file"].as<std::string>());
      float n_pos_corr = 0.f;
      float n_uas_corr = 0.f;
      float n_las_corr = 0.f;
      float n_total = 0.f;
      while (std::getline(ifs, buffer)) {
        boost::algorithm::trim(buffer);
        if (buffer.empty()) {
          if (pos_engine != nullptr) {
            pos_engine->postag(tokens, postags);
          } else {
            postags.resize(gold_postags.size());
            for (unsigned i = 0; i < gold_postags.size(); ++i) {
              postags[i] = gold_postags[i];
            }
          }
          if (par_engine != nullptr) {
            par_engine->predict(tokens, postags, heads, deprels);
          }

          boost::algorithm::trim(header);
          if (!header.empty()) {
            std::cout << header << "\n";
          }
          for (unsigned i = 0; i < tokens.size(); ++i) {
            std::cout << i + 1 << "\t" << tokens[i] << "\t_\t";
            if (pos_engine == nullptr) {
              std::cout << gold_postags[i] << "\t_\t_\t";
            } else {
              std::cout << postags[i] << "\t_\tGoldPOS=" << gold_postags[i] << "\t";
            }
            if (par_engine == nullptr) {
              std::cout << "_\t_\t_\t_\n";
            } else {
              std::cout << heads[i] << "\t" << deprels[i] << "\t_\t_\n";
            }
            if (load_postag_model && postags[i] == gold_postags[i]) {
              n_pos_corr += 1.;
            }
            if (load_parse_model && heads[i] == gold_heads[i]) {
              n_uas_corr += 1.; 
              if (deprels[i] == gold_deprels[i]) { n_las_corr += 1.; }
            }
            n_total += 1.;
          }
          std::cout << "\n";

          tokens.clear();
          header = "";
          if (load_postag_model || load_parse_model) {
            postags.clear(); gold_postags.clear();
          }
          if (load_parse_model) {
            heads.clear(); gold_heads.clear();
            deprels.clear(); gold_deprels.clear();
          }
        } else if (buffer[0] == '#') {
          header += "\n" + buffer;
        } else {
          std::vector<std::string> data;
          boost::algorithm::split(data, buffer, boost::is_any_of("\t "));
          tokens.push_back(data[1]);
          if (load_postag_model || load_parse_model) {
            gold_postags.push_back(data[3]);
          }
          if (load_parse_model) {
            gold_heads.push_back(data[6] == "_" ? 0 : boost::lexical_cast<unsigned>(data[6]));
            gold_deprels.push_back(data[7]);
          }
        }
      }
      if (load_postag_model) {
        _INFO << "[evaluate] postag accuracy: " << n_pos_corr / n_total;
      }
      if (load_parse_model) {
        _INFO << "[evaluate] UAS accuracy: " << n_uas_corr / n_total;
        _INFO << "[evaluate] LAS accuracy: " << n_las_corr / n_total;
      }
    }
  }
  return 0;
}
