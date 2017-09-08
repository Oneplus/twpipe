#ifndef __TWPIPE_MATH_H__
#define __TWPIPE_MATH_H__

#include <vector>
#include <random>

namespace twpipe {

struct Math {
  static void softmax_inplace(std::vector<float>& x);

  static unsigned distribution_sample(const std::vector<float>& prob,
                                      std::mt19937& gen);
};

}

#endif  //  end for __TWPIPE_MATH_H__