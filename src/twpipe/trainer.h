#ifndef __TWPIPE_TRAINER_H__
#define __TWPIPE_TRAINER_H__

#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct Trainer {
  std::string model_name;
  unsigned max_iter;
  unsigned evaluate_stops;
  unsigned evaluate_skips;
  float lambda_;

  static po::options_description get_options();

  Trainer(const po::variables_map & conf);

  bool need_evaluate(unsigned iter, unsigned n_trained);
};

}

#endif  //  end if __TWPIPE_TRAINER_H__