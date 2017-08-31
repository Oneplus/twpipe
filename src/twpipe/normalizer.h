#ifndef __TWPIPE_NORMALIZER_H__
#define __TWPIPE_NORMALIZER_H__

#include <boost/regex.hpp>

namespace twpipe {

struct GloveNormalizer {
  static boost::regex url_regex;
  static boost::regex user_regex;
  static boost::regex smile_regex;
  static boost::regex lolface_regex;
  static boost::regex sadface_regex;
  static boost::regex neuralface_regex;
  static boost::regex heart_regex;
  static boost::regex number_regex;
  static boost::regex repeat_regex;
  static boost::regex elong_regex;

  // dealing with username, url, emoticon, expressive lengthening
  // match with the glove normalization process.
  static std::string normalize(const std::string & word);
};

struct OwoputiNormalizer {
  static std::string punct_chars;
  static std::string entity;
  static std::string url_start_1;
  static std::string common_tlds;
  static std::string cc_tlds;
  static std::string url_start_2;
  static std::string url_body;
  static std::string url_extra_crap_before_end;
  static std::string url_end;
  static std::string bound;
  static std::string email;
  static std::string url;
  static std::string at_signs_chars;
  static std::string valid_mention_or_list;
  static boost::regex url_regex;
  static boost::regex mention_regex;

  static std::string normalize(const std::string & word);
};

}

#endif  //  end for __TWPIPE_NORMALIZER_H__