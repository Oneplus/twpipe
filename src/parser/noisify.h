#ifndef NOISIFY_H
#define NOISIFY_H

#include "corpus.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

struct Noisifier {
  enum NOISIFY_METHOD { kNone, kSingletonDroput, kWordDropout};
  NOISIFY_METHOD noisify_method;
  float singleton_dropout_prob;
  unsigned unk;
  Corpus& corpus;

  static po::options_description get_options();

  Noisifier(const po::variables_map& conf, Corpus& corpus);

  void noisify(InputUnits& units) const;

  void denoisify(InputUnits& units) const;
};


#endif  //  end for NOISIFY_H