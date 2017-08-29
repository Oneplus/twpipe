#include "tokenize_model_builder.h"
#include "lin_rnn_tokenize_model.h"
#include "seg_rnn_tokenize_model.h"
#include "twpipe/logging.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

template<class RNNBuilderType> const unsigned LinearRNNTokenizeModel<RNNBuilderType>::kB = 0;
template<class RNNBuilderType> const unsigned LinearRNNTokenizeModel<RNNBuilderType>::kI = 1;
template<class RNNBuilderType> const unsigned LinearRNNTokenizeModel<RNNBuilderType>::kO = 2;

template<> const char* LinearGRUTokenizeModel::name = "LinearGRUTokenizeModel";
template<> const char* LinearLSTMTokenizeModel::name = "LinearLSTMTokenizeModel";
template<> const char* SegmentalGRUTokenizeModel::name = "SegmentalGRUTokenizeModel";
template<> const char* SegmentalLSTMTokenizeModel::name = "SegmentalLSTMTokenizeModel";

TokenizeModelBuilder::TokenizeModelBuilder(po::variables_map & conf) {
  model_name = conf["tok-model-name"].as<std::string>();

  if (model_name == "bi-gru") {
    model_type = kLinearGRUTokenizeModel;
  } else if (model_name == "bi-lstm") {
    model_type = kLinearLSTMTokenizeModel;
  } else if (model_name == "seg-gru") {
    model_type = kSegmentalGRUTokenizeModel;
  } else if (model_name == "seg-lstm") {
    model_type = kSegmentalLSTMTokenizeModel;
  } else {
    _ERROR << "[tokenize|model_builder] unknown tokenize model: " << model_name;
  }

  char_size = AlphabetCollection::get()->char_map.size();
  char_dim = (conf.count("tok-char-dim") ? conf["tok-char-dim"].as<unsigned>() : 0);
  hidden_dim = (conf.count("tok-hidden-dim") ? conf["tok-hidden-dim"].as<unsigned>() : 0);
  n_layers = (conf.count("tok-n-layer") ? conf["tok-n-layer"].as<unsigned>() : 0);
  seg_dim = (conf.count("tok-seg-dim") ? conf["tok-seg-dim"].as<unsigned>() : 0);
  dur_dim = (conf.count("tok-dur-dim") ? conf["tok-dur-dim"].as<unsigned>() : 0);
}

TokenizeModel * TokenizeModelBuilder::build(dynet::ParameterCollection & model) {
  TokenizeModel * engine = nullptr;
  if (model_type == kLinearGRUTokenizeModel) {
    engine = new LinearGRUTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers);
  } else if (model_type == kLinearLSTMTokenizeModel) {
    engine = new LinearLSTMTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers);
  } else if (model_type == kSegmentalGRUTokenizeModel) {
    engine = new SegmentalGRUTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers,
                                           seg_dim, dur_dim);
  } else if (model_type == kSegmentalLSTMTokenizeModel) {
    engine = new SegmentalLSTMTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers,
                                            seg_dim, dur_dim);
  } else {
    _ERROR << "[tokenize|model_builder] Unknown tokenize model: " << model_name;
    exit(1);
  }

  return engine;
}

void TokenizeModelBuilder::to_json() {
  Model::get()->to_json(Model::kTokenizerName, {
    { "name", model_name },
    { "n-chars", boost::lexical_cast<std::string>(char_size) },
    { "char-dim", boost::lexical_cast<std::string>(char_dim) },
    { "hidden-dim", boost::lexical_cast<std::string>(hidden_dim) },
    { "n-layers", boost::lexical_cast<std::string>(n_layers) },
    { "seg-dim", boost::lexical_cast<std::string>(seg_dim) },
    { "dur-dim", boost::lexical_cast<std::string>(dur_dim) }
  });
}

TokenizeModel * TokenizeModelBuilder::from_json(dynet::ParameterCollection & model) {
  TokenizeModel * engine = nullptr;

  Model * globals = Model::get();
  model_name = globals->from_json(Model::kTokenizerName, "name");
  unsigned temp_size = 
    boost::lexical_cast<unsigned>(globals->from_json(Model::kTokenizerName, "n-chars"));
  if (char_size == 0) {
    char_size = temp_size;
  } else {
    BOOST_ASSERT_MSG(char_size == temp_size, "[tokenize|model_builder] char-size mismatch!");
  }
  char_dim = 
    boost::lexical_cast<unsigned>(globals->from_json(Model::kTokenizerName, "char-dim"));
  hidden_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kTokenizerName, "hidden-dim"));
  n_layers =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kTokenizerName, "n-layers"));

  if (model_name == "bi-gru") {
    model_type = kLinearGRUTokenizeModel;
    engine = new LinearGRUTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers);
  } else if (model_name == "bi-lstm") {
    model_type = kLinearLSTMTokenizeModel;
    engine = new LinearLSTMTokenizeModel(model, char_size, char_dim, hidden_dim, n_layers);
  } else if (model_name == "seg-gru") {
    model_type = kSegmentalGRUTokenizeModel;
    BOOST_ASSERT_MSG(false, "SegmentalRNN not implemented.");
  } else if (model_name == "seg-lstm") {
    model_type = kSegmentalLSTMTokenizeModel;
    BOOST_ASSERT_MSG(false, "SegmentalRNN not implemented.");
  } else {
    _ERROR << "[tokenize|model_builder] unknow tokenize model: " << model_name;
    exit(1);
  }

  globals->from_json(Model::kTokenizerName, model);
  return engine;
}

}
