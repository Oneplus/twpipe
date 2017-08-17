#include "tokenize_model.h"
#include <set>

po::options_description twpipe::TokenizeModel::get_options() {
  po::options_description model_opts("Tokenizer model options");
  model_opts.add_options()
    ("tok-model-name", po::value<std::string>()->default_value("bi-gru"), "the model name [bi-gru|bi-lstm|seg-rnn]")
    ("tok-char-dim", po::value<unsigned>()->default_value(64), "the charactor embedding dimension.")
    ("tok-hidden-dim", po::value<unsigned>()->default_value(64), "the hidden dimension of rnn.")
    ("tok-n-layer", po::value<unsigned>()->default_value(1), "the number of layers.")
    ("tok-seg-dim", po::value<unsigned>()->default_value(64), "the dimension of segment (only used in seg-rnn).")
    ;
  return model_opts;
}

twpipe::TokenizeModel::TokenizeModel(dynet::ParameterCollection & model,
                                     const Alphabet & char_map) : 
  model(model),
  char_map(char_map),
  space_cid(char_map.get(Corpus::SPACE)) {
}

void twpipe::TokenizeModel::tokenize(const std::string & input) {
  std::vector<std::string> result;
  tokenize(input, result);

  std::cout << "# text = " << input << "\n";
  for (unsigned i = 0; i < result.size(); ++i) {
    std::cout << i + 1 << "\t"
      << result[i] << "\t" << "_\t"
      << "_\t" << "_\t"
      << "_\t"
      << "_\t" << "_\t"
      << "_\t" << "_\n";
  }
  std::cout << "\n";
}

void twpipe::TokenizeModel::tokenize(const std::string & input, std::vector<std::string> & result) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  decode(input, result);
}

std::tuple<float, float, float> twpipe::TokenizeModel::evaluate(const std::vector<std::string>& gold,
                                                                const std::vector<std::string>& predict) {
  std::set<std::pair<unsigned, unsigned>> gold_boundaries;
  std::set<std::pair<unsigned, unsigned>> pred_boundaries;
  unsigned i = 0;
  for (const std::string & word : gold) {
    unsigned len = word.size();
    gold_boundaries.insert(std::make_pair(i, len));
    i += word.size();
  }
  i = 0;
  for (const std::string & word : predict) {
    unsigned len = word.size();
    pred_boundaries.insert(std::make_pair(i, len));
    i += word.size();
  }
  float n_recall = 0;
  for (auto boundary : pred_boundaries) {
    if (gold_boundaries.find(boundary) != gold_boundaries.end()) { n_recall += 1.; }
  }
  return std::tuple<float, float, float>(n_recall, static_cast<float>(predict.size()), static_cast<float>(gold.size()));
}
