#include "tokenize_model.h"
#include "twpipe/alphabet_collection.h"
#include <set>

po::options_description twpipe::AbstractTokenizeModel::get_options() {
  po::options_description model_opts("Tokenizer model options");
  model_opts.add_options()
    ("tok-model-name", po::value<std::string>()->default_value("bi-gru"), "the model name [bi-gru|bi-lstm|seg-rnn]")
    ("tok-char-dim", po::value<unsigned>()->default_value(64), "the character embedding dimension.")
    ("tok-hidden-dim", po::value<unsigned>()->default_value(64), "the hidden dimension of rnn.")
    ("tok-n-layer", po::value<unsigned>()->default_value(1), "the number of layers.")
    ("tok-seg-dim", po::value<unsigned>()->default_value(64), "the dimension of segment (only used in seg-rnn).")
    ;
  return model_opts;
}

twpipe::AbstractTokenizeModel::AbstractTokenizeModel(dynet::ParameterCollection & model) :
  model(model),
  space_cid(AlphabetCollection::get()->char_map.get(Corpus::SPACE)) {
}

std::tuple<float, float, float> twpipe::AbstractTokenizeModel::fscore(const std::vector<std::string>& gold,
                                                                      const std::vector<std::string>& predict) {
  std::set<std::pair<unsigned, unsigned>> gold_boundaries;
  std::set<std::pair<unsigned, unsigned>> pred_boundaries;
  unsigned i = 0;
  for (const std::string & word : gold) {
    unsigned len = word.size();
    gold_boundaries.insert(std::make_pair(i, len));
    i += len;
  }
  i = 0;
  for (const std::string & word : predict) {
    unsigned len = word.size();
    pred_boundaries.insert(std::make_pair(i, len));
    i += len;
  }
  float n_recall = 0;
  for (auto boundary : pred_boundaries) {
    if (gold_boundaries.find(boundary) != gold_boundaries.end()) { n_recall += 1.; }
  }
  return {n_recall, static_cast<float>(predict.size()), static_cast<float>(gold.size())};
}

twpipe::TokenizeModel::TokenizeModel(dynet::ParameterCollection &model) : AbstractTokenizeModel(model) {

}

void twpipe::TokenizeModel::tokenize(const std::string &input) {
  std::vector<std::string> result;
  tokenize(input, result);

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

void twpipe::TokenizeModel::tokenize(const std::string &input, std::vector<std::string> & result) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  decode(input, result);
}


std::tuple<float, float, float> twpipe::TokenizeModel::evaluate(const Instance & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  std::vector<std::string> result;
  decode(inst.raw_sentence, result);

  std::vector<std::string> gold;
  for (unsigned i = 1; i < inst.input_units.size(); ++i) {
    gold.push_back(inst.input_units[i].word);
  }

  return fscore(gold, result);
};

twpipe::SentenceSegmentAndTokenizeModel::SentenceSegmentAndTokenizeModel(dynet::ParameterCollection &model)
  : AbstractTokenizeModel(model) {

}

void twpipe::SentenceSegmentAndTokenizeModel::sentsegment_and_tokenize(const std::string &input) {
  std::vector<std::vector<std::string>> result;
  sentsegment_and_tokenize(input, result);

  for (unsigned s = 0; s < result.size(); ++s) {
    std::cout << "# text = " << input << "\n";
    std::cout << "# sid=" << s + 1 << "\n";
    for (unsigned i = 0; i < result.size(); ++i) {
      std::cout << i + 1 << "\t"
                << result[s][i] << "\t" << "_\t"
                << "_\t" << "_\t"
                << "_\t"
                << "_\t" << "_\t"
                << "_\t" << "_\n";
    }
    std::cout << "\n";
  }
}

void twpipe::SentenceSegmentAndTokenizeModel::sentsegment_and_tokenize(const std::string &input,
                                                                       std::vector<std::vector<std::string>> &result) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  decode(input, result);
}

std::tuple<float, float, float> twpipe::SentenceSegmentAndTokenizeModel::evaluate(const Instance & inst) {
  dynet::ComputationGraph cg;
  new_graph(cg);
  std::vector<std::vector<std::string>> result;
  decode(inst.raw_sentence, result);

  std::vector<std::string> predict;
  for (const auto & r : result) {
    for (const auto & w : r) { predict.push_back(w);}
  }
  std::vector<std::string> gold;
  for (unsigned i = 1; i < inst.input_units.size(); ++i) {
    gold.push_back(inst.input_units[i].word);
  }

  return fscore(gold, predict);
};
