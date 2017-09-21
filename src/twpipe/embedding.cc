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
  size_t found = embedding_file.find("glove");
  normalizer_type = kNone;
  if (found != std::string::npos) { normalizer_type = kGlove; }
  pretrained[Corpus::BAD0] = std::vector<float>(dim, 0.f);
  pretrained[Corpus::UNK] = std::vector<float>(dim, 0.f);
  pretrained[Corpus::ROOT] = std::vector<float>(dim, 0.f);
  _INFO << "[embedding] loading from " << embedding_file << " with " << dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  // get the header in word2vec styled embedding.
  std::getline(ifs, line);
  std::vector<float> v(dim, 0.f);
  std::string word;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> word;
    // actually, there should be a checking about the embedding dimension.
    for (unsigned i = 0; i < dim; ++i) { iss >> v[i]; }
    pretrained[word] = v;
  }
  std::string normalizer_type_name = "none";
  if (normalizer_type == kGlove) { normalizer_type_name = "glove"; }
  _INFO << "[embedding] normalizer type: " << normalizer_type_name;
  _INFO << "[embedding] loaded embedding " << pretrained.size() << " entries.";
}

void WordEmbedding::empty(unsigned dim) {
  dim_ = dim;
  normalizer_type = kNone;
  pretrained[Corpus::BAD0] = std::vector<float>(dim, 0.f);
  pretrained[Corpus::UNK] = std::vector<float>(dim, 0.f);
  pretrained[Corpus::ROOT] = std::vector<float>(dim, 0.f);
  _INFO << "[embedding] loaded embedding " << pretrained.size() << " entries.";
}

void WordEmbedding::render(const std::vector<std::string>& words,
                           std::vector<std::vector<float>>& values) {
  values.clear();
  for (const auto & word : words) {
    std::string normalized_word = word;
    if (normalizer_type == kGlove) {
      normalized_word = GloveNormalizer::normalize(normalized_word);
    }
    auto it = pretrained.find(normalized_word);
    values.push_back(it == pretrained.end() ?
                     std::vector<float>(dim_, 0.f) :
                     it->second);
  }
}

unsigned WordEmbedding::dim() {
  return dim_;
}

}