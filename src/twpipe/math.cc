#include "math.h"

void twpipe::Math::softmax_inplace(std::vector<float>& x) {
  float m = x[0];
  for (const float& _x : x) { m = (_x > m ? _x : m); }
  float s = 0.;
  for (unsigned i = 0; i < x.size(); ++i) {
    x[i] = exp(x[i] - m);
    s += x[i];
  }
  for (unsigned i = 0; i < x.size(); ++i) { x[i] /= s; }
}

unsigned twpipe::Math::distribution_sample(const std::vector<float>& prob,
                                           std::mt19937 & gen) {
  std::discrete_distribution<unsigned> distrib(prob.begin(), prob.end());
  return distrib(gen);
}
