#include "state.h"
#include "corpus.h"


State::State(unsigned n) : heads(n, Corpus::BAD_HED), deprels(n, Corpus::BAD_DEL) {
}

float State::loss(const std::vector<unsigned>& gold_heads,
                  const std::vector<unsigned>& gold_deprels) {
  BOOST_ASSERT_MSG(gold_heads.size() == heads.size(),
    "# of heads should be equal to # of gold heads");
  BOOST_ASSERT_MSG(gold_deprels.size() == deprels.size(),
    "# of deprels should be equal to # of gold deprels");
  float n_corr = 0., n_total = 0.;
  for (unsigned i = 0; i < gold_heads.size(); ++i) {
    if (gold_heads[i] == heads[i] && gold_deprels[i] == deprels[i])
      n_corr++;
    n_total++;
  }
  return (n_corr / n_total);
}

bool State::terminated() const {
  return !(stack.size() > 2 || buffer.size() > 1);
}
