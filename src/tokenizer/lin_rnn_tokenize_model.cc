#include "lin_rnn_tokenize_model.h"
#include "twpipe/unicode.h"

twpipe::LinearTokenizeModel::LinearTokenizeModel(dynet::ParameterCollection &model) :
  TokenizeModel(model),
  one_more_space_regex("[ ]{2,}") {

}

void twpipe::LinearTokenizeModel::get_gold_labels(const twpipe::Instance &inst,
                                                  const std::string &clean_input,
                                                  std::vector<unsigned> &labels) {
  auto & input_units = inst.input_units;

  unsigned len = 0;
  unsigned j = 1, k = 0; // j start from 1 because the first one is dummy root.
  for (unsigned i = 0; i < clean_input.size(); i += len) {
    len = utf8_len(clean_input[i]);
    std::string ch = clean_input.substr(i, len);
    unsigned lid = (ch == " " ? kO : (k == 0 ? kB : kI));
    labels.push_back(lid);
    if (ch != " ") {
      ++k;
      if (k == input_units[j].cids.size()) { k = 0; ++j; }
    }
  }
}

twpipe::LinearSentenceSegmentAndTokenizeModel::LinearSentenceSegmentAndTokenizeModel(dynet::ParameterCollection &model) :
  SentenceSegmentAndTokenizeModel(model),
  one_more_space_regex("[ ]{2,}") {

}

void twpipe::LinearSentenceSegmentAndTokenizeModel::get_colored(const std::vector<std::vector<unsigned>> &tree,
                                                              unsigned now, unsigned target,
                                                              std::vector<unsigned> &colors) {
  colors[now] = target;
  for (unsigned c : tree[now]) {
    get_colored(tree, c, target, colors);
  }
}

void twpipe::LinearSentenceSegmentAndTokenizeModel::get_colored(const twpipe::Instance &inst,
                                                              std::vector<unsigned> &colors) {
  auto & parse_units = inst.parse_units;
  std::vector<std::vector<unsigned>> tree(parse_units.size());
  std::vector<unsigned> roots;
  colors.resize(parse_units.size());

  for (unsigned j = 1; j < parse_units.size(); ++j) {
    unsigned head = parse_units[j].head;
    tree[head].push_back(j);
    if (head == 0) { roots.push_back(j); }
  }

  for (unsigned root : roots) {
    get_colored(tree, root, root, colors);
  }
}

void twpipe::LinearSentenceSegmentAndTokenizeModel::get_gold_labels(const twpipe::Instance &inst,
                                                                  const std::string &clean_input,
                                                                  std::vector<unsigned> &labels) {
  auto & input_units = inst.input_units;
  std::vector<unsigned> colors;
  get_colored(inst, colors);

  unsigned len = 0;
  unsigned j = 1, k = 0; // j start from 1 because the first one is dummy root.
  for (unsigned i = 0; i < clean_input.size(); i += len) {
    len = utf8_len(clean_input[i]);
    std::string ch = clean_input.substr(i, len);
    unsigned lid = (ch == " " ? kO : (k == 0 ? (j == 1 || colors[j - 1] != colors[j] ? kB1 : kB) : kI));
    labels.push_back(lid);
    if (ch != " ") {
      ++k;
      if (k == input_units[j].cids.size()) { k = 0; ++j; }
    }
  }
}

void twpipe::CharactersTokenizeModel::get_chars(const std::string &clean_input, std::vector<unsigned> &cids,
                                                twpipe::Alphabet &char_map, std::vector<std::string> *chars) {
  unsigned len = 0;
  for (unsigned i = 0; i < clean_input.size(); i += len) {
    len = utf8_len(clean_input[i]);
    std::string ch = clean_input.substr(i, len);
    if (chars != nullptr) { chars->push_back(ch); }
    unsigned cid = (char_map.contains(ch) ? char_map.get(ch) : char_map.get(Corpus::UNK));
    cids.push_back(cid);
  }
}

void twpipe::CharactersTokenizeModel::get_chars_and_char_categories(const std::string &clean_input,
                                                                    std::vector<unsigned> &cids,
                                                                    std::vector<unsigned> &ctids,
                                                                    twpipe::Alphabet &char_map,
                                                                    std::vector<std::string> *chars) {
  unsigned len = 0;
  for (unsigned i = 0; i < clean_input.size(); i += len) {
    len = utf8_len(clean_input[i]);
    std::string ch = clean_input.substr(i, len);
    char32_t unicode_chr = utf8_to_unicode_first_(ch);
    uint8_t category = ufal::unilib::unicode::compact_category(unicode_chr);

    if (chars != nullptr) { chars->push_back(ch); }
    unsigned cid = (char_map.contains(ch) ? char_map.get(ch) : char_map.get(Corpus::UNK));
    cids.push_back(cid);
    ctids.push_back(category);
  }
}
