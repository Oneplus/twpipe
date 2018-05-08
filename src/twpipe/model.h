#ifndef __TWPIPE_MODEL_H__
#define __TWPIPE_MODEL_H__

#include <iostream>
#include <boost/program_options.hpp>
#include "dynet/model.h"
#include "alphabet.h"
#include "json.hpp"

namespace po = boost::program_options;

namespace twpipe {

typedef std::pair<std::string, std::string> StrConfigItemType;
typedef std::pair<std::string, unsigned> IntConfigItemType;

class Model {
protected:
  nlohmann::json payload;
  static Model * instance;

  Model();

public:
  static const char* kGeneral;
  static const char* kTokenizerName;
  static const char* kSentenceSegmentAndTokenizeName;
  static const char* kPostaggerName;
  static const char* kParserName;

  static po::options_description get_options();

  static Model * get();

  void save(const std::string & filename);

  void load(const std::string & filename);

  void to_json(const std::string & phase_name,
               const std::vector<StrConfigItemType> & str_conf);

  void to_json(const std::string & name,
               const Alphabet & alphabet);

  void to_json(const std::string & phase_name,
               dynet::ParameterCollection & model);

  std::string from_json(const std::string & phase_name, const std::string & key);

  void from_json(const std::string & name,
                 Alphabet & alphabet);

  void from_json(const std::string & phase_name,
                 dynet::ParameterCollection & model);

  bool has_segmentor_and_tokenizer_model() const;

  bool has_tokenizer_model() const;

  bool has_postagger_model() const;

  bool has_parser_model() const;

  bool valid_phase_name(const std::string & phase_name);
};

}

#endif  //  end for __TWPIPE_MODEL_H__