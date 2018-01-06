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

  unsigned id;
  std::vector<unsigned> categories;
  std::vector<std::vector<float>> probs;

  EnsembleInstance(unsigned id,
                   std::vector<unsigned> & categories,
                   std::vector<std::vector<float>> & probs);
};

typedef std::vector<EnsembleInstance> EnsembleInstances;

struct EnsembleUtils {
  static void load_ensemble_instances(const std::string & path,
                                      EnsembleInstances & instances);
};

}

#endif  //  end for __TWPIPE_ENSEMBLE_H__