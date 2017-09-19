#ifndef __TWPIPE_EMBEDDING_H__
#define __TWPIPE_EMBEDDING_H__

#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "alphabet.h"

namespace po = boost::program_options;

namespace twpipe {

struct WordEmbedding {
protected:
  enum NORMALIZER_TYPE { kNone, kGlove };
  static WordEmbedding * instance;
  std::unordered_map<std::string, std::vector<float>> pretrained;
  NORMALIZER_TYPE normalizer_type;
  unsigned dim_;

  WordEmbedding();

public:
  static po::options_description get_options();

  static WordEmbedding* get();

  void load(const std::string& embedding_file, unsigned dim);

  void empty(unsigned dim);

  void render(const std::vector<std::string> & words,
              std::vector<std::vector<float>> & values);

  unsigned dim();
};

}

#endif // !__TWPIPE_EMBEDDING_H__
