#ifndef __TWPIPE_ENSEMBLE_H__
#define __TWPIPE_ENSEMBLE_H__

#include <string>
#include <vector>
#include <unordered_map>

namespace twpipe {

struct EnsembleInstance {
  static const char* id_name;
  static const char* category_name;
  static const char* prob_name;

  std::vector<unsigned> categories;
  std::vector<std::vector<float>> probs;

  EnsembleInstance(std::vector<unsigned> & categories,
                   std::vector<std::vector<float>> & probs);
};

typedef std::unordered_map<unsigned, EnsembleInstance> EnsembleInstances;

struct EnsembleUtils {
  static void load_ensemble_instances(const std::string & path,
                                      EnsembleInstances & instances);
};

}

#endif  //  end for __TWPIPE_ENSEMBLE_H__