#ifndef __TWPIPE_PARSER_SAMPLER_H__
#define __TWPIPE_PARSER_SAMPLER_H__

#include <vector>
#include <boost/program_options.hpp>
#include "parse_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct Sampler {
  virtual void sample(const std::vector<std::string> & words,
                      const std::vector<std::string> & postags,
                      const std::vector<unsigned> & heads,
                      const std::vector<std::string> & deprels,
                      std::vector<unsigned> & actions) = 0;
};

struct OracleSampler: public Sampler {
  ParseModel * engine;

  OracleSampler(ParseModel * engine);

  void sample(const std::vector<std::string> & words,
              const std::vector<std::string> & postags,
              const std::vector<unsigned> & heads,
              const std::vector<std::string> & deprels,
              std::vector<unsigned> & actions) override;
};

struct VanillaSampler: public Sampler {
  ParseModel * engine;

  VanillaSampler(ParseModel * engine);

  void sample(const std::vector<std::string> & words,
              const std::vector<std::string> & postags,
              const std::vector<unsigned> & heads,
              const std::vector<std::string> & deprels,
              std::vector<unsigned> & actions) override ;
};

struct EnsembleSampler: public Sampler {
  std::vector<ParseModel *>& engines;

  EnsembleSampler(std::vector<ParseModel *>& engines);

  void sample(const std::vector<std::string> & words,
              const std::vector<std::string> & postags,
              const std::vector<unsigned> & heads,
              const std::vector<std::string> & deprels,
              std::vector<unsigned> & actions) override ;
};

}

#endif  //  end for __TWPIPE_PARSER_SAMPLER_H__