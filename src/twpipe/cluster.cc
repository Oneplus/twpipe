#include "cluster.h"
#include "logging.h"
#include "corpus.h"
#include "normalizer.h"
#include <iostream>
#include <fstream>

namespace twpipe {

WordCluster* WordCluster::instance = nullptr;

WordCluster::WordCluster() {
}

po::options_description WordCluster::get_options() {
  po::options_description opts("Embedding options");
  opts.add_options()
    ("cluster", po::value<std::string>(), "the path to the cluster file.")
    ;
  return opts;
}

WordCluster * WordCluster::get() {
  if (instance == nullptr) {
    instance = new WordCluster();
  }
  return instance;
}

void WordCluster::load(const std::string & cluster_file) {
  cluster[Corpus::BAD0] = Corpus::BAD0;
  cluster[Corpus::UNK] = Corpus::UNK;
  cluster[Corpus::ROOT] = Corpus::ROOT;
  _INFO << "[cluster] loading from " << cluster_file;
  std::ifstream ifs(cluster_file);
  BOOST_ASSERT_MSG(ifs, "Failed to load embedding file.");
  std::string line;
  std::string word;
  std::string type;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    iss >> type;
    iss >> word;
    cluster[word] = type;
  }
  _INFO << "[cluster] loaded cluster " << cluster.size() << " entries.";
}

void WordCluster::empty() {
  cluster[Corpus::BAD0] = Corpus::BAD0;
  cluster[Corpus::UNK] = Corpus::UNK;
  cluster[Corpus::ROOT] = Corpus::ROOT;
  _INFO << "[cluster] loaded cluster " << cluster.size() << " entries.";
}

void WordCluster::render(const std::vector<std::string>& words, 
                         std::vector<std::string>& values) {
  values.clear();
  for (const auto & word : words) {
    auto it = cluster.find(OwoputiNormalizer::normalize(word));
    values.push_back(it == cluster.end() ? Corpus::UNK : it->second);
  }
}

}