#ifndef __TWPIPE_PARSER_NOISIFY_H__
#define __TWPIPE_PARSER_NOISIFY_H__

#include "twpipe/corpus.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct Noisifier {
  enum NOISIFY_METHOD { kNone, kSingletonDroput, kWordDropout };
  NOISIFY_METHOD noisify_method;
  Corpus& corpus;
  float singleton_dropout_prob;
  unsigned unk;

  Noisifier(Corpus & corpus,
            const std::string & noisify_method_name,
            float singleton_dropout_prob = 0.5);

  void noisify(InputUnits& units) const;

  void denoisify(InputUnits& units) const;
};

}

#endif  //  end for NOISIFY_H