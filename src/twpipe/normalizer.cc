#include "normalizer.h"
#include <boost/algorithm/string.hpp>

namespace twpipe {

std::string _repeat(const boost::smatch & what) {
  return what[1].str();
}

std::string _elong(const boost::smatch & what) {
  return what[1].str() + what[2].str();
}

boost::regex GloveNormalizer::url_regex("https?:\\/\\/(\\S+|www\\.(\\w+\\.)+\\S*)");
boost::regex GloveNormalizer::user_regex("@\\w+");
boost::regex GloveNormalizer::smile_regex("[8:=;]['`\\\\-]?[)d]+|[)d]+['`\\\\-]?[8:=;]");
boost::regex GloveNormalizer::lolface_regex("[8:=;]['`\\\\-]?p+");
boost::regex GloveNormalizer::sadface_regex("[8:=;]['`\\\\-]?\\(+|\\)+['`\\\\-]?[8:=;]");
boost::regex GloveNormalizer::neuralface_regex("[8:=;]['`\\\\-]?[\\/|l*]");
boost::regex GloveNormalizer::number_regex("^[-+]?[.\\d]*[\\d]+[:,.\\d]*$");
boost::regex GloveNormalizer::heart_regex("<3");
boost::regex GloveNormalizer::repeat_regex("([!?.]){2,}+");
boost::regex GloveNormalizer::elong_regex("(\\S*?)(\\w)\\2{2,}");

std::string GloveNormalizer::normalize(const std::string & word) {
  std::string ret = word;
  ret = boost::regex_replace(ret, url_regex, "<url>");
  ret = boost::regex_replace(ret, user_regex, "<user>");
  ret = boost::regex_replace(ret, smile_regex, "<smile>");
  ret = boost::regex_replace(ret, lolface_regex, "<lolface>");
  ret = boost::regex_replace(ret, sadface_regex, "<sadface>");
  ret = boost::regex_replace(ret, neuralface_regex, "<neutralface>");
  ret = boost::regex_replace(ret, heart_regex, "<heart>");
  ret = boost::regex_replace(ret, number_regex, "<number>");
  ret = boost::regex_replace(ret, repeat_regex, _repeat, boost::match_default | boost::format_all);
  ret = boost::regex_replace(ret, elong_regex, _elong);
  boost::to_lower(ret);
  return ret;
}

std::string OwoputiNormalizer::punct_chars("['\"¡°¡±¡®¡¯.?!¡­,:;]");
std::string OwoputiNormalizer::entity("&(?:amp|lt|gt|quot);");
std::string OwoputiNormalizer::url_start_1("(?:https?://|\bwww\\.)");
std::string OwoputiNormalizer::common_tlds("(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)");
std::string OwoputiNormalizer::cc_tlds("(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|"
                                       "bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|"
                                       "er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|"
                                       "hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|"
                                       "lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|"
                                       "nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|"
                                       "sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|"
                                       "va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)");
std::string OwoputiNormalizer::url_start_2("\\b(?:[A-Za-z\\d-])+(?:\\.[A-Za-z0-9]+){0,3}\\.(?:" + common_tlds + "|" + cc_tlds + ")(?:\\." + cc_tlds + ")?(?=\\W|$)");
std::string OwoputiNormalizer::url_body("(?:[^\\.\\s<>][^\\s<>]*?)?");
std::string OwoputiNormalizer::url_end("(?:\\.\\.+|[<>]|\\s|$)");
std::string OwoputiNormalizer::url_extra_crap_before_end("(?:" + punct_chars + "|" + entity + ")+?");
std::string OwoputiNormalizer::url("(?:" + url_start_1 + "|" + url_start_2 + ")" + url_body + "(?=(?:" + url_extra_crap_before_end + ")?" + url_end + ")");
std::string OwoputiNormalizer::valid_mention_or_list("([^a-z0-9_!#$%&*@]|^|(?:^|[^a-z0-9_+~.-])RT:?)([@]+)([a-z0-9_]{1,20})(/[a-z][a-z0-9_\\-]{0,24})?");

boost::regex OwoputiNormalizer::url_regex(url);
boost::regex OwoputiNormalizer::mention_regex(valid_mention_or_list);

std::string OwoputiNormalizer::normalize(const std::string & word) {
  std::string ret = boost::to_lower_copy(word);
  if (boost::regex_match(ret.c_str(), mention_regex)) {
    return "<@MENTION>";
  }
  return ret;
}

}