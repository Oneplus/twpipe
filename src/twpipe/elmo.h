#ifndef __TWPIPE_ELMO_H__
#define __TWPIPE_ELMO_H__

#include <vector>
#include <unordered_map>
#include <boost/program_options.hpp>
#include "alphabet.h"

namespace po = boost::program_options;

namespace twpipe {

struct ELMo {
protected:
  static ELMo * instance;
  std::unordered_map<std::string, std::vector<std::vector<float>>> pretrained;
  unsigned dim_;

  ELMo();

public:
  static po::options_description get_options();

  static ELMo* get();

  void load(const std::string& embedding_file, unsigned dim);

  void empty(unsigned dim);

  void render(const std::vector<std::string> & words,
              std::vector<std::vector<float>> & values);

  unsigned dim();
};

}

#endif // !__TWPIPE_ELMO_H__
