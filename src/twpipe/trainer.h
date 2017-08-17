#ifndef __TWPIPE_TRAINER_H__
#define __TWPIPE_TRAINER_H__

#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace twpipe {

struct Trainer {
  std::string model_name;
  unsigned max_iter;
  unsigned early_stop;

  static po::options_description get_options();

  Trainer(const po::variables_map & conf);
};

}

#endif  //  end if __TWPIPE_TRAINER_H__