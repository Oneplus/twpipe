#include "postag_model_builder.h"
#include "char_rnn_postag_model.h"
#include "char_rnn_crf_postag_model.h"
// #include "char_rnn_wcluster_postag_model.h"
#include "word_rnn_postag_model.h"
#include "word_char_rnn_postag_model.h"
#include "twpipe/logging.h"
#include "twpipe/model.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

template<> const char* CharacterGRUPostagModel::name = "CharacterGRUPostagModel";
template<> const char* CharacterLSTMPostagModel::name = "CharacterLSTMPostagModel";
template<> const char* CharacterGRUCRFPostagModel::name = "CharacterGRUCRFPostagModel";
template<> const char* CharacterLSTMCRFPostagModel::name = "CharacterLSTMCRFPostagModel";
// template<> const char* CharacterGRUWithClusterPostagModel::name = "CharacterGRUWithClusterPostagModel";
// template<> const char* CharacterLSTMWithClusterPostagModel::name = "CharacterLSTMWithClusterPostagModel";
template<> const char* WordGRUPostagModel::name = "WordGRUPostagModel";
template<> const char* WordLSTMPostagModel::name = "WordLSTMPostagModel";
template<> const char* WordCharacterGRUPostagModel::name = "WordCharacterGRUPostagModel";
template<> const char* WordCharacterLSTMPostagModel::name = "WordCharacterLSTMPostagModel";

PostagModelBuilder::PostagModelBuilder(po::variables_map & conf) {
  if (conf.count("pos-model-name")) {
    model_name = conf["pos-model-name"].as<std::string>();
    model_type = get_model_type(model_name);
  }

  word_size = AlphabetCollection::get()->word_map.size();
  char_size = AlphabetCollection::get()->char_map.size();
  pos_size = AlphabetCollection::get()->pos_map.size();
  word_dim = (conf.count("pos-word-dim") ? conf["pos-word-dim"].as<unsigned>() : 0);
  char_dim = (conf.count("pos-char-dim") ? conf["pos-char-dim"].as<unsigned>() : 0);
  char_hidden_dim = (conf.count("pos-char-hidden-dim") ? conf["pos-char-hidden-dim"].as<unsigned>() : 0);
  char_n_layers = (conf.count("pos-char-n-layer") ? conf["pos-char-n-layer"].as<unsigned>() : 0);
  word_hidden_dim = (conf.count("pos-word-hidden-dim") ? conf["pos-word-hidden-dim"].as<unsigned>() : 0);
  word_n_layers = (conf.count("pos-word-n-layer") ? conf["pos-word-n-layer"].as<unsigned>() : 0);
  cluster_dim = (conf.count("pos-cluster-dim") ? conf["pos-cluster-dim"].as<unsigned>() : 0);
  cluster_hidden_dim = (conf.count("pos-cluster-hidden-dim") ? conf["pos-cluster-hidden-dim"].as<unsigned>() : 0);
  cluster_n_layers = (conf.count("pos-cluster-n-layer") ? conf["pos-cluster-n-layer"].as<unsigned>() : 0);
  pos_dim = (conf.count("pos-pos-dim") ? conf["pos-pos-dim"].as<unsigned>() : 0);
  embed_dim = (conf.count("embedding-dim") ? conf["embedding-dim"].as<unsigned>() : 0);
}

PostagModel * PostagModelBuilder::build(dynet::ParameterCollection & model) {
  PostagModel * engine = nullptr;
  if (model_type == kCharacterGRUPostagModel) {
    engine = new CharacterGRUPostagModel(model, char_size, char_dim, char_hidden_dim,
                                         char_n_layers, embed_dim, word_hidden_dim,
                                         word_n_layers, pos_dim);
  } else if (model_type == kCharacterLSTMPostagModel) {
    engine = new CharacterLSTMPostagModel(model, char_size, char_dim, char_hidden_dim,
                                          char_n_layers, embed_dim, word_hidden_dim,
                                          word_n_layers, pos_dim);
  } else if (model_type == kCharacterGRUPostagCRFModel) {
    engine = new CharacterGRUCRFPostagModel(model, char_size, char_dim, char_hidden_dim,
                                            char_n_layers, embed_dim, word_hidden_dim,
                                            word_n_layers, pos_dim);
  } else if (model_type == kCharacterLSTMPostagCRFModel) {
    engine = new CharacterLSTMCRFPostagModel(model, char_size, char_dim, char_hidden_dim,
                                             char_n_layers, embed_dim, word_hidden_dim,
                                             word_n_layers, pos_dim);
  } else if (model_type == kCharacterClusterGRUPostagModel) {
    /*engine = new CharacterGRUWithClusterPostagModel(model, char_size, char_dim, char_hidden_dim,
                                                    char_n_layers, embed_dim, word_hidden_dim,
                                                    word_n_layers, cluster_dim, cluster_hidden_dim,
                                                    cluster_n_layers, pos_dim);*/
  } else if (model_type == kCharacterClusterLSTMPostagModel) {
    /*engine = new CharacterLSTMWithClusterPostagModel(model, char_size, char_dim, char_hidden_dim,
                                                     char_n_layers, embed_dim, word_hidden_dim,
                                                     word_n_layers, cluster_dim, cluster_hidden_dim,
                                                     cluster_n_layers, pos_dim);*/
  } else if (model_type == kWordGRUPostagModel) {
    engine = new WordGRUPostagModel(model, word_size, word_dim, embed_dim, 
                                    word_hidden_dim, word_n_layers, pos_dim);
  } else if (model_type == kWordLSTMPostagModel) {
    engine = new WordLSTMPostagModel(model, word_size, word_dim, embed_dim,
                                     word_hidden_dim, word_n_layers, pos_dim);
  } else if (model_type == kWordCharacterGRUPostagModel) {
    engine = new WordCharacterGRUPostagModel(model, char_size, char_dim, char_hidden_dim,
                                             char_n_layers, word_size, word_dim, embed_dim,
                                             word_hidden_dim, word_n_layers, pos_dim);
  } else if (model_type == kWordCharacterLSTMPostagModel) {
    engine = new WordCharacterLSTMPostagModel(model, char_size, char_dim, char_hidden_dim,
                                              char_n_layers, word_size, word_dim, embed_dim,
                                              word_hidden_dim, word_n_layers, pos_dim);
  } else {
    _ERROR << "[postag|model_builder] unknow postag model: " << model_name;
    exit(1);
  }
  return engine;
}

void PostagModelBuilder::to_json() {
  Model::get()->to_json(Model::kPostaggerName, {
    {"name", model_name},
    {"word-hidden-dim", std::to_string(word_hidden_dim)},
    {"word-n-layers", std::to_string(word_n_layers)},
    {"pos-dim", std::to_string(pos_dim)},
    {"n-postags", std::to_string(pos_size)},
    {"emb-dim", std::to_string(embed_dim)}
  });

  if (model_type == kCharacterGRUPostagModel ||
      model_type == kCharacterLSTMPostagModel ||
      model_type == kCharacterGRUPostagCRFModel ||
      model_type == kCharacterLSTMPostagCRFModel ||
      model_type == kCharacterClusterGRUPostagModel ||
      model_type == kCharacterClusterLSTMPostagModel) {
    Model::get()->to_json(Model::kPostaggerName, {
      { "n-chars", std::to_string(char_size) },
      { "char-dim", std::to_string(char_dim) },
      { "char-hidden-dim", std::to_string(char_hidden_dim) },
      { "char-n-layers", std::to_string(char_n_layers) }
    });
  } else {
    Model::get()->to_json(Model::kPostaggerName, {
      { "n-words", std::to_string(word_size) },
      { "word-dim", std::to_string(word_dim) },
    });
  }

  if (model_type == kCharacterClusterGRUPostagModel ||
      model_type == kCharacterClusterLSTMPostagModel) {
    Model::get()->to_json(Model::kPostaggerName, {
      {"cluster-dim", std::to_string(cluster_dim)},
      {"cluster-hidden-dim", std::to_string(cluster_hidden_dim)},
      {"cluster-n-layers", std::to_string(cluster_n_layers)},
    });
  }
}

PostagModel * PostagModelBuilder::from_json(dynet::ParameterCollection & model) {
  PostagModel * engine = nullptr;

  Model * globals = Model::get();
  model_name = globals->from_json(Model::kPostaggerName, "name");
  unsigned temp_size;

  model_type = get_model_type(model_name);

  if (model_type == kCharacterGRUPostagModel ||
      model_type == kCharacterLSTMPostagModel ||
      model_type == kCharacterGRUPostagCRFModel ||
      model_type == kCharacterLSTMPostagCRFModel ||
      model_type == kCharacterClusterGRUPostagModel ||
      model_type == kCharacterClusterLSTMPostagModel ||
      model_type == kWordCharacterGRUPostagModel ||
      model_type == kWordCharacterLSTMPostagModel) {
    temp_size = 
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "n-chars"));
    if (char_size == 0) {
      char_size = temp_size;
    } else {
      BOOST_ASSERT_MSG(char_size == temp_size, "[postag|model_builder] char-size mismatch!");
    }

    char_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "char-dim"));
    char_hidden_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "char-hidden-dim"));
    char_n_layers =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "char-n-layers"));
  }
  if (model_type == kWordCharacterGRUPostagModel ||
      model_type == kWordCharacterLSTMPostagModel ||
      model_type == kWordLSTMPostagModel ||
      model_type == kWordGRUPostagModel){
    temp_size = 
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "n-words"));
    if (word_size == 0) {
      word_size = temp_size;
    } else {
      BOOST_ASSERT_MSG(word_size == temp_size, "[postag|model_builder] word-size mismatch!");
    }

    word_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "word-dim"));   
  }

  temp_size =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "n-postags"));
  if (pos_size == 0) {
    pos_size = temp_size;
  } else {
    BOOST_ASSERT_MSG(pos_size == temp_size, "[postag|model_builder] pos-size mismatch!");
  }

  word_hidden_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "word-hidden-dim"));
  word_n_layers =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "word-n-layers")); 
  pos_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "pos-dim"));
  embed_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "emb-dim"));

  if (model_type == kCharacterClusterGRUPostagModel || model_type == kCharacterClusterLSTMPostagModel) {
    cluster_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "cluster-dim"));
    cluster_hidden_dim =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "cluster-hidden-dim"));
    cluster_n_layers =
      boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "cluster-n-layers"));
  }

  engine = build(model);
  globals->from_json(Model::kPostaggerName, model);
  
  return engine;
}

PostagModelBuilder::ModelType PostagModelBuilder::get_model_type(const std::string & model_name) {
  ModelType ret;
  if (model_name == "char-gru") {
    ret = kCharacterGRUPostagModel;
  } else if (model_name == "char-lstm") {
    ret = kCharacterLSTMPostagModel;
  } else if (model_name == "char-gru-crf") {
    ret = kCharacterGRUPostagCRFModel;
  } else if (model_name == "char-lstm-crf") {
    ret = kCharacterLSTMPostagCRFModel;
  } else if (model_name == "char-gru-wcluster") {
    ret = kCharacterClusterGRUPostagModel;
  } else if (model_name == "char-lstm-wcluster") {
    ret = kCharacterClusterLSTMPostagModel;
  } else if (model_name == "word-gru") {
    ret = kWordGRUPostagModel;
  } else if (model_name == "word-lstm") {
    ret = kWordLSTMPostagModel;
  } else if (model_name == "word-char-gru") {
    ret = kWordCharacterGRUPostagModel;
  } else if (model_name == "word-char-lstm") {
    ret = kWordCharacterLSTMPostagModel;
  } else {
    _ERROR << "[postag|model_builder] unknown postag model: " << model_name;
    exit(1);
  }

  return ret;
}

}
