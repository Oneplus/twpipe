#include "embedding.h"
#include "logging.h"
#include "corpus.h"
#include "normalizer.h"
#include <fstream>

namespace twpipe {

WordEmbedding * WordEmbedding::instance = nullptr;

WordEmbedding::WordEmbedding() {
}

po::options_description WordEmbedding::get_options() {
  po::options_description embed_opts("Embedding options");
  embed_opts.add_options()
    ("embedding", po::value<std::string>(), "the path to the embedding file.")
    ("embedding-dim", po::value<unsigned>()->default_value(100), "the dimension of embedding.")
    ;
  return embed_opts;
}

WordEmbedding * WordEmbedding::get() {
  if (instance == nullptr) {
    instance = new WordEmbedding();
  }
  return instance;
}

void WordEmbedding::load(const std::string & embedding_file, unsigned dim) {
  dim_ = dim;
  pretrained[Corpus::BAD0] = std::vector<float>(dim, 0.);
  pretrained[Corpus::UNK] = std::vector<float>(dim, 0.);
  pretrained[Corpus::ROOT] = std::vector<float>(dim, 0.);
  _INFO << "[embedding] loading from " << embedding_file << " with " << dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  // get the header in word2vec styled embedding.
  std::getline(ifs, line);
  std::vector<float> v(dim, 0.);
  std::string word;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> word;
    // actually, there should be a checking about the embedding dimension.
    for (unsigned i = 0; i < dim; ++i) { iss >> v[i]; }
    pretrained[word] = v;
  }
  _INFO << "[embedding] loaded embedding " << pretrained.size() << " entries.";
}

void WordEmbedding::empty(unsigned dim) {
  pretrained[Corpus::BAD0] = std::vector<float>(dim, 0.);
  pretrained[Corpus::UNK] = std::vector<float>(dim, 0.);
  pretrained[Corpus::ROOT] = std::vector<float>(dim, 0.);
  _INFO << "[embedding] loaded embedding " << pretrained.size() << " entries.";
}

void WordEmbedding::render(const std::vector<std::string>& words,
                           std::vector<std::vector<float>>& values) {
  values.clear();
  for (const auto & word : words) {
    auto it = pretrained.find(GloveNormalizer::normalize(word));
    values.push_back(it == pretrained.end() ?
                     std::vector<float>(dim_, 0.) :
                     it->second);
  }
}

unsigned WordEmbedding::dim() {
  return dim_;
}

}