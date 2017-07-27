#include "postag_model_builder.h"
#include "twpipe/logging.h"
#include "twpipe/model.h"

namespace twpipe {

template<> const char* CharacterGRUPostagModel::name = "CharacterGRUPostagModel";
template<> const char* CharacterLSTMPostagModel::name = "CharacterLSTMPostagModel";

PostagModelBuilder::PostagModelBuilder(po::variables_map & conf,
                                       const Alphabet & char_map,
                                       const Alphabet & pos_map) :
  char_map(char_map),
  pos_map(pos_map) {
  model_name = conf["pos-model-name"].as<std::string>();

  if (model_name == "char-gru") {
    model_type = kCharacterGRUPostagModel;
  } else if (model_name == "char-lstm") {
    model_type = kCharacterLSTMPostagModel;
  } else {
    _ERROR << "[postag|model_builder] unknow postaag model: " << model_name;
  }

  char_size = char_map.size();
  pos_size = pos_map.size();
  char_dim = (conf.count("pos-char-dim") ? conf["pos-char-dim"].as<unsigned>() : 0);
  char_hidden_dim = (conf.count("pos-char-hidden-dim") ? conf["pos-char-hidden-dim"].as<unsigned>() : 0);
  char_n_layers = (conf.count("pos-char-n-layer") ? conf["pos-char-n-layer"].as<unsigned>() : 0);
  word_dim = (conf.count("pos-word-dim") ? conf["pos-word-dim"].as<unsigned>() : 0);
  word_hidden_dim = (conf.count("pos-word-hidden-dim") ? conf["pos-word-hidden-dim"].as<unsigned>() : 0);
  word_n_layers = (conf.count("pos-word-n-layer") ? conf["pos-word-n-layer"].as<unsigned>() : 0);
  pos_dim = (conf.count("pos-pos-dim") ? conf["pos-pos-dim"].as<unsigned>() : 0);
  embed_dim = (conf.count("embedding-dim") ? conf["embedding-dim"].as<unsigned>() : 0);
}

PostagModel * PostagModelBuilder::build(dynet::ParameterCollection & model) {
  PostagModel * engine = nullptr;
  if (model_type == kCharacterGRUPostagModel) {
    engine = new CharacterGRUPostagModel(model, char_size, char_dim, char_hidden_dim,
                                         char_n_layers, embed_dim, word_dim, word_hidden_dim,
                                         word_n_layers, pos_dim, char_map, pos_map);
  } else if (model_type == kCharacterLSTMPostagModel) {
    engine = new CharacterLSTMPostagModel(model, char_size, char_dim, char_hidden_dim,
                                          char_n_layers, embed_dim, word_dim, word_hidden_dim,
                                          word_n_layers, pos_dim, char_map, pos_map);
  }
  return engine;
}

void PostagModelBuilder::to_json() {
  Model::get()->to_json(Model::kPostaggerName, {
    {"name", model_name},
    {"n-chars", boost::lexical_cast<std::string>(char_size)},
    {"char-dim", boost::lexical_cast<std::string>(char_dim)},
    {"char-hidden-dim", boost::lexical_cast<std::string>(char_hidden_dim)},
    {"char-n-layers", boost::lexical_cast<std::string>(char_n_layers)},
    {"word-dim", boost::lexical_cast<std::string>(word_dim)},
    {"word-hidden-dim", boost::lexical_cast<std::string>(word_hidden_dim)},
    {"word-n-layers", boost::lexical_cast<std::string>(word_n_layers)},
    {"pos-dim", boost::lexical_cast<std::string>(pos_dim)},
    {"n-postags", boost::lexical_cast<std::string>(pos_size)},
    {"emb-dim", boost::lexical_cast<std::string>(embed_dim)}
  });
}

PostagModel * PostagModelBuilder::from_json(dynet::ParameterCollection & model) {
  PostagModel * engine = nullptr;

  Model * globals = Model::get();
  model_name = globals->from_json(Model::kPostaggerName, "name");
  unsigned temp_size;
  temp_size = 
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "n-chars"));
  if (char_size == 0) {
    char_size = temp_size;
  } else {
    BOOST_ASSERT_MSG(char_size == temp_size, "[postag|model_builder] pos-size mismatch!");
  }
  char_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "char-dim"));
  char_hidden_dim =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "hidden-dim"));
  char_n_layers =
    boost::lexical_cast<unsigned>(globals->from_json(Model::kPostaggerName, "char-n-layers"));
  return engine;
}

}
