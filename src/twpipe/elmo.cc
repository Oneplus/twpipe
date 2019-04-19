#include "elmo.h"
#include "logging.h"
#include "normalizer.h"
#include <fstream>
#include <boost/algorithm/string.hpp>

namespace twpipe {

ELMo * ELMo::instance = nullptr;

ELMo::ELMo(): dim_(0) {
}

po::options_description ELMo::get_options() {
  po::options_description embed_opts("ELMo options");
  embed_opts.add_options()
    ("elmo", po::value<std::string>(), "the path to the embedding file.")
    ("elmo-dim", po::value<unsigned>()->default_value(1024), "the dimension of embedding.")
    ;
  return embed_opts;
}

ELMo * ELMo::get() {
  if (instance == nullptr) {
    instance = new ELMo();
  }
  return instance;
}

void ELMo::load(const std::string & embedding_file, unsigned dim) {
  dim_ = dim;
  _INFO << "[elmo] loading from " << embedding_file << " with " << dim << " dimensions.";
  std::ifstream ifs(embedding_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  std::string word;

  int cnt = 0;
  while (true) {
    std::getline(ifs, line);
    std::string key = line;
    boost::algorithm::trim(key);
    if (key.empty()) {
      break;
    }

    auto & values = pretrained[key];
    values.clear();
    std::vector<float> v(dim, 0.f);
    while (true) {
      std::getline(ifs, line);
      boost::algorithm::trim(line);
      if (line.empty()) {
        break;
      }
      std::istringstream iss(line);
      // actually, there should be a checking about the embedding dimension.
      for (unsigned i = 0; i < dim; ++i) { iss >> v[i]; }
      values.push_back(v);
    }
    cnt ++;
    if (cnt % 1000 == 0) {
      _INFO << "[elmo] loaded " << cnt
      << " sentences, " << pretrained.size() << " entries.";
    }
  }
  _INFO << "[elmo] loaded " << cnt
        << " sentences, " << pretrained.size() << " entries.";
}

void ELMo::empty(unsigned dim) {
  dim_ = dim;
  _INFO << "[elmo] loaded embedding " << pretrained.size() << " entries.";
}

void ELMo::render(const std::vector<std::string>& words,
  std::vector<std::vector<float>>& values) {
  std::string key;
  for (const auto & word : words) {
    if (key.empty()) {
      key = word;
    } else {
      key.append("\t");
      key.append(word);
    }
  }
  boost::algorithm::replace_all(key, ".", "$period$");
  boost::algorithm::replace_all(key, "/", "$backslash$");
  auto it = pretrained.find(key);
  if (it == pretrained.end()) {
    _ERROR << "[elmo] key \"" << key << "\" not founded.";
    for (int i = 0; i < words.size(); ++i) {
      values.emplace_back(std::vector<float>(dim_, 0.f));
    }
  } else {
    for (const auto & val : it->second) {
      values.push_back(val);
    }
  }
}

unsigned ELMo::dim() {
  return dim_;
}

}
