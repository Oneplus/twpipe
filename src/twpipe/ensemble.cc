#include "ensemble.h"
#include "json.hpp"
#include <iostream>
#include <fstream>

namespace twpipe {

const char* EnsembleInstance::id_name = "id";
const char* EnsembleInstance::category_name = "category";
const char* EnsembleInstance::prob_name = "prob";

EnsembleInstance::EnsembleInstance(std::vector<unsigned>& categories,
                                   std::vector<std::vector<float>>& probs) :
  categories(categories),
  probs(probs) {
}

void EnsembleUtils::load_ensemble_instances(const std::string & path,
                                            EnsembleInstances & instances) {
  nlohmann::json payload;
  std::ifstream ifs(path);
  std::string buffer;
  while (std::getline(ifs, buffer)) {
    payload = nlohmann::json::parse(buffer.begin(), buffer.end());
    unsigned id = payload.at(EnsembleInstance::id_name).get<unsigned>();
    std::vector<unsigned> actions = payload.at(EnsembleInstance::category_name).get<std::vector<unsigned>>();
    std::vector<std::vector<float>> probs = payload.at(EnsembleInstance::prob_name).get<std::vector<std::vector<float>>>();

    instances.insert(std::make_pair(id, EnsembleInstance(actions, probs)));
  }
}

}
