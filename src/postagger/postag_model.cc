#include "postag_model.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

po::options_description PostagModel::get_options() {
  po::options_description model_opts("Postagger model options");
  model_opts.add_options()
    ("pos-model-name", po::value<std::string>()->default_value("char-gru"), "the model name [char-gru|char-lstm|word-gru|word-lstm|char-gru-crf|word-lstm-crf].")
    ("pos-char-dim", po::value<unsigned>()->default_value(32), "the character embedding dimension.")
    ("pos-char-hidden-dim", po::value<unsigned>()->default_value(32), "the hidden dimension of char-rnn.")
    ("pos-char-n-layer", po::value<unsigned>()->default_value(1), "the number of layers of char-rnn.")
    ("pos-word-dim", po::value<unsigned>()->default_value(64), "the character embedding dimension.")
    ("pos-word-hidden-dim", po::value<unsigned>()->default_value(64), "the hidden dimension of word-rnn.")
    ("pos-word-n-layer", po::value<unsigned>()->default_value(2), "the number of layers of word-rnn.")
    ("pos-cluster-dim", po::value<unsigned>()->default_value(8), "the cluster bit embedding dimension.")
    ("pos-cluster-n-layer", po::value<unsigned>()->default_value(1), "the number of layers for cluster-rnn.")
    ("pos-cluster-hidden-dim", po::value<unsigned>()->default_value(8), "the hidden dimension of cluster-nn.")
    ("pos-pos-dim", po::value<unsigned>()->default_value(16), "the dimension of postag.")
    ;
  return model_opts;
}

PostagModel::PostagModel(dynet::ParameterCollection & model) :
  model(model),
  pos_size(AlphabetCollection::get()->pos_map.size()) {
}

void PostagModel::postag(const std::vector<std::string>& words) {
  std::vector<std::string> tags;
  postag(words, tags);
  
  for (unsigned i = 0; i < words.size(); ++i) {
    std::cout << i + 1 << "\t" << words[i] << "\t_\t"
      << tags[i] << "\t_\t_\t_\t_\t_\n";
  }
  std::cout << "\n";
}

void PostagModel::postag(const std::vector<std::string>& words,
                         std::vector<std::string>& tags) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  decode(words, tags);
}

std::pair<float, float> PostagModel::evaluate(const std::vector<std::string>& gold,
                                              const std::vector<std::string>& prediction) {
  BOOST_ASSERT_MSG(gold.size() == prediction.size(), "");
  float n_recall = 0., n_gold = 0.;
  for (unsigned i = 0; i < gold.size(); ++i) {
    if (gold[i] == prediction[i]) { n_recall += 1.; }
  }
  n_gold = static_cast<float>(gold.size());
  return std::pair<float, float>(n_recall, n_gold);
}

}