#include "model.h"
#include <fstream>
#include <boost/algorithm/string.hpp>

namespace twpipe {

const char* Model::kGeneral = "general";
const char* Model::kTokenizerName = "tokenizer";
const char* Model::kPostaggerName = "postagger";
const char* Model::kParserName = "parser";

Model* Model::instance = nullptr;

Model::Model() {
  payload[kTokenizerName] = nullptr;
  payload[kPostaggerName] = nullptr;
  payload[kParserName] = nullptr;
}

po::options_description Model::get_options() {
  po::options_description model_opts("Model options");
  model_opts.add_options()
    ("model", po::value<std::string>(), "model file")
    ;
  return model_opts;
}

Model * Model::get() {
  if (!instance) {
    instance = new Model;
  }
  return instance;
}

void Model::save(const std::string & filename) {
  std::ofstream ofs(filename);
  BOOST_ASSERT_MSG(ofs, "[model] failed to open file.");
  ofs << payload;
}

void Model::load(const std::string & filename) {
  std::ifstream ifs(filename);
  BOOST_ASSERT_MSG(ifs, "[model] failed to open file.");
  ifs >> payload;
}

void Model::to_json(const std::string & phase_name,
                    const std::vector<StrConfigItemType>& str_conf) {
  if (!valid_phase_name(phase_name)) {
    BOOST_ASSERT_MSG(false, "[model] invalid phase name.");
  }

  auto & json = payload[phase_name]["config"];
  for (auto & conf : str_conf) { json[conf.first] = conf.second; }
}

void Model::to_json(const std::string & name,
                    const Alphabet & alphabet) {
  auto & json = payload[kGeneral][name];
  const auto & str_to_id = alphabet.str_to_id;
  for (const auto & it : str_to_id) {
    json[it.first] = it.second;
  }
}

void Model::to_json(const std::string & phase_name,
                    dynet::ParameterCollection & model) {
  if (!valid_phase_name(phase_name)) {
    BOOST_ASSERT_MSG(false, "[model] invalid phase name.");
  }

  const dynet::ParameterCollectionStorage & storage = model.get_storage();
  auto & json = payload[phase_name]["model"];
  for (auto & p : storage.params) { 
    json[p->name]["dim"] = p->dim.size();
    json[p->name]["value"] = dynet::as_vector(p->values);
  }
  for (auto & p : storage.lookup_params) {
    json[p->name]["dim"] = p->all_dim.size();
    json[p->name]["value"] = dynet::as_vector(p->all_values);
  }
}

std::string Model::from_json(const std::string & phase_name,
                             const std::string & key) {
  if (!valid_phase_name(phase_name)) {
    BOOST_ASSERT_MSG(false, "[model] invalid phase name.");
  }

  auto & json = payload[phase_name]["config"];
  return json.value(key, "__empty__");
}

void Model::from_json(const std::string & name, Alphabet & alphabet) {
  auto & json = payload[kGeneral][name];
  for (auto it = json.begin(); it != json.end(); ++it) {
    alphabet.insert(it.key(), it.value());
  }
}

void Model::from_json(const std::string & phase_name,
                      dynet::ParameterCollection & model) {
  if (!valid_phase_name(phase_name)) {
    BOOST_ASSERT_MSG(false, "[model] invalid phase name.");
  }

  const dynet::ParameterCollectionStorage & storage = model.get_storage();
  auto & json = payload[phase_name]["model"];
  for (auto & p : storage.params) {
    unsigned dim = json[p->name]["dim"];
    BOOST_ASSERT_MSG(p->dim.size() == dim, "[model] mismatch dimension when loading.");
    std::vector<float> values(dim);
    values = json[p->name]["value"].get<std::vector<float>>();
    dynet::TensorTools::set_elements(p->values, values);
  }
  for (auto & p : storage.lookup_params) {
    unsigned dim = json[p->name]["dim"];
    BOOST_ASSERT_MSG(p->all_dim.size() == dim, "[model] mismatch dimension when loading.");
    std::vector<float> values(dim);
    values = json[p->name]["value"].get<std::vector<float>>();
    dynet::TensorTools::set_elements(p->all_values, values);
  }
}

bool Model::has_tokenizer_model() const {
  return !payload[kTokenizerName].is_null();
}

bool Model::has_postagger_model() const {
  return !payload[kPostaggerName].is_null();
}

bool Model::has_parser_model() const {
  return !payload[kParserName].is_null();
}

bool Model::valid_phase_name(const std::string & phase_name) {
  if (phase_name == kTokenizerName ||
      phase_name == kPostaggerName ||
      phase_name == kParserName) {
    return true;
  }
  return false;
}

}
