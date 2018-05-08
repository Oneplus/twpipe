#ifndef __TWPIPE_PARSER_TESTER_H__
#define __TWPIPE_PARSER_TESTER_H__

#include <vector>
#include <boost/program_options.hpp>
#include "parse_model.h"

namespace po = boost::program_options;

namespace twpipe {

struct Tester {
  virtual void test(const std::vector<std::string> & words,
                    const std::vector<std::string> & postags,
                    const std::vector<unsigned> & heads,
                    const std::vector<std::string> & deprels,
                    const std::vector<unsigned> & actions,
                    std::vector<std::vector<float>> & prob) = 0;
};

struct OracleTester: public Tester {
  ParseModel * engine;

  OracleTester(ParseModel * engine);

  void test(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            const std::vector<unsigned> & heads,
            const std::vector<std::string> & deprels,
            const std::vector<unsigned> & actions,
            std::vector<std::vector<float>> & prob) override;
};

struct VanillaTester: public Tester {
  ParseModel * engine;

  VanillaTester(ParseModel * engine);

  void test(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            const std::vector<unsigned> & heads,
            const std::vector<std::string> & deprels,
            const std::vector<unsigned> & actions,
            std::vector<std::vector<float>> & prob) override ;
};

struct EnsembleTester: public Tester {
  std::vector<ParseModel *>& engines;

  EnsembleTester(std::vector<ParseModel *>& engines);

  void test(const std::vector<std::string> & words,
            const std::vector<std::string> & postags,
            const std::vector<unsigned> & heads,
            const std::vector<std::string> & deprels,
            const std::vector<unsigned> & actions,
            std::vector<std::vector<float>> & prob) override ;
};

}

#endif  //  end for __TWPIPE_PARSER_TESTER_H__