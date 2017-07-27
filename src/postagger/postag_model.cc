#include "postag_model.h"

namespace twpipe {

po::options_description PostagModel::get_options() {
  po::options_description model_opts("Postagger model options");
  model_opts.add_options()
    ("pos-model-name", po::value<std::string>()->default_value("char-gru"), "the model name [char-gru|char-lstm].")
    ("pos-char-dim", po::value<unsigned>()->default_value(64), "the character embedding dimension.")
    ("pos-char-hidden-dim", po::value<unsigned>()->default_value(64), "the hidden dimension of char-rnn.")
    ("pos-char-n-layer", po::value<unsigned>()->default_value(1), "the number of character lstm layers.")
    ("pos-word-dim", po::value<unsigned>()->default_value(64), "the character embedding dimension.")
    ("pos-word-hidden-dim", po::value<unsigned>()->default_value(64), "the hidden dimension of char-rnn.")
    ("pos-word-n-layer", po::value<unsigned>()->default_value(1), "the number of word lstm layers.")
    ("pos-pos-dim", po::value<unsigned>()->default_value(32), "the dimension of postag.")
    ;
  return model_opts;
}

PostagModel::PostagModel(dynet::ParameterCollection & model, 
                         const Alphabet & pos_map) :
  model(model), 
  pos_map(pos_map),
  pos_size(pos_map.size()) {
}

std::pair<float, float> PostagModel::evaluate(const std::vector<std::string>& gold,
                                              const std::vector<std::string>& prediction) {
  BOOST_ASSERT_MSG(gold.size() == prediction.size(), "");
  float n_recall = 0., n_gold = 0.;
  for (unsigned i = 0; i < gold.size(); ++i) {
    if (gold[i] == prediction[i]) { n_recall += 1.; }
  }
  return std::pair<float, float>(n_recall, n_gold);
}

}