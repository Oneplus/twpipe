#include "arceager.h"
#include "logging.h"
#include "corpus.h"

ArcEager::ArcEager(const Alphabet& map) : TransitionSystem(map) {
  n_actions = 3 + 2 * map.size();

  action_names.push_back("SHIFT");  // 0
  action_names.push_back("DROP");   // 1
  action_names.push_back("REDUCE"); // 2
  for (unsigned i = 0; i < map.size(); ++i) {
    action_names.push_back("LEFT-" + map.get(i));
    action_names.push_back("RIGHT-" + map.get(i));
  }
  _INFO << "TransitionSystem:: show action names:";
  for (const auto& action_name : action_names) {
    _INFO << "- " << action_name;
  }
}

std::string ArcEager::name(unsigned id) const {
  BOOST_ASSERT_MSG(id < action_names.size(), "id in illegal range");
  return action_names[id];
}

bool ArcEager::allow_nonprojective() const {
  return false;
}

unsigned ArcEager::num_actions() const { return n_actions; }

unsigned ArcEager::num_deprels() const { return deprel_map.size(); }

bool ArcEager::is_shift(const unsigned& action) { return action == 0; }
bool ArcEager::is_drop(const unsigned& action) { return action == 1; }
bool ArcEager::is_reduce(const unsigned& action) { return action == 2; }
bool ArcEager::is_left(const unsigned& action) { return (action > 2 && action % 2 == 1); }
bool ArcEager::is_right(const unsigned& action) { return (action > 2 && action % 2 == 0); }

unsigned ArcEager::get_shift_id() { return 0; }
unsigned ArcEager::get_drop_id() { return 1; }
unsigned ArcEager::get_reduce_id() { return 2; }
unsigned ArcEager::get_left_id(const unsigned& deprel)  { return deprel * 2 + 3; }
unsigned ArcEager::get_right_id(const unsigned& deprel) { return deprel * 2 + 4; }

void ArcEager::shift_unsafe(State& state) const {
  state.stack.push_back(state.buffer.back());
  state.buffer.pop_back();
}

void ArcEager::drop_unsafe(State& state) const {
  unsigned b = state.buffer.back();
  state.heads[b] = Corpus::REMOVED_HED;
  state.deprels[b] = Corpus::REMOVED_DEL;
  state.buffer.pop_back();
}

void ArcEager::left_unsafe(State& state, const unsigned& deprel) const {
  unsigned mod = state.stack.back();
  unsigned hed = state.buffer.back();
  state.stack.pop_back();
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

void ArcEager::right_unsafe(State& state, const unsigned& deprel) const {
  unsigned hed = state.stack.back();
  unsigned mod = state.buffer.back();
  state.stack.push_back(state.buffer.back());
  state.buffer.pop_back();
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

void ArcEager::reduce_unsafe(State& state) const {
  state.stack.pop_back();
}

float ArcEager::shift_dynamic_loss_unsafe(State& state,
                                          const std::vector<unsigned>& ref_heads,
                                          const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned b = state.buffer.back();
  for (unsigned i = 1; i < state.stack.size(); ++i) {
    unsigned k = state.stack[i];
    if (ref_heads[k] == b && state.heads[k] == Corpus::BAD_HED) { c += 1.; }
    if (ref_heads[b] == k) { c += 1.; }
  }
  shift_unsafe(state);
  return c;
}

float ArcEager::left_dynamic_loss_unsafe(State& state,
                                         const unsigned& deprel,
                                         const std::vector<unsigned>& ref_heads,
                                         const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned s = state.stack.back();
  unsigned b = state.buffer.back();
  for (unsigned i = 1; i < state.buffer.size() - 1; ++i) {
    unsigned k = state.buffer[i];
    if (ref_heads[k] == s) { c += 1.; }
    if (ref_heads[s] == k) { c += 1.; }
  }
  if (ref_heads[b] == s) { c += 1.; }
  if (ref_heads[s] == b && ref_deprels[s] != deprel) { c += 1.; }
  left_unsafe(state, deprel);
  return c;
}

float ArcEager::right_dynamic_loss_unsafe(State& state,
                                          const unsigned& deprel,
                                          const std::vector<unsigned>& ref_heads,
                                          const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned s = state.stack.back();
  unsigned b = state.buffer.back();
  for (unsigned i = 1; i < state.stack.size() - 1; ++i) {
    unsigned k = state.stack[i];
    if (ref_heads[k] == b && state.heads[k] == Corpus::BAD_HED) { c += 1.; }
    if (ref_heads[b] == k) { c += 1.; }
  }
  if (ref_heads[s] == b && state.heads[s] == Corpus::BAD_HED) { c += 1.; }

  /*for (unsigned i = 1; i < state.buffer.size() - 1; ++i) {
    unsigned k = state.buffer[i];
    if (ref_heads[b] == k) { c += 2.; }
  }*/
  if (ref_heads[b] > b) { c += 1.; }
  if (ref_heads[b] == s && ref_deprels[b] != deprel) { c += 1.; }
  right_unsafe(state, deprel);
  return c;
}

float ArcEager::reduce_dynamic_loss_unsafe(State& state,
                                           const std::vector<unsigned>& ref_heads,
                                           const std::vector<unsigned>& ref_deprels) const {
  float c = 0.;
  unsigned s = state.stack.back();
  for (unsigned i = 1; i < state.buffer.size(); ++i) {
    unsigned k = state.buffer[i];
    if (ref_heads[k] == s) { c += 1.; }
  }
  reduce_unsafe(state);
  return c;
}

void ArcEager::get_transition_costs(const State& state,
                                    const std::vector<unsigned>& actions,
                                    const std::vector<unsigned>& ref_heads,
                                    const std::vector<unsigned>& ref_deprels,
                                    std::vector<float>& rewards) {
  float wrong_left = -1e8;
  float wrong_right = -1e8;
  rewards.clear();

  for (unsigned act : actions) {
    State next_state(state);
    if (is_shift(act)) {
      rewards.push_back(-shift_dynamic_loss_unsafe(next_state, ref_heads, ref_deprels));
    } else if (is_left(act)) {
      unsigned deprel = parse_label(act);
      unsigned hed = state.buffer.back(), mod = state.stack.back();
      if (ref_heads[mod] == hed && ref_deprels[mod] == deprel) {
        // assume that actions are unique and there is only one correct left action.
        rewards.push_back(-left_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels));
      } else if (wrong_left == -1e8) {
        wrong_left = -left_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels);
        rewards.push_back(wrong_left);
      } else {
        rewards.push_back(wrong_left);
      }
    } else if (is_right(act)) {
      unsigned deprel = parse_label(act);
      unsigned mod = state.stack.back(), hed = state.stack[state.stack.size() - 2];
      if (ref_heads[mod] == hed && ref_deprels[mod] == deprel) {
        rewards.push_back(-right_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels));
      } else if (wrong_right == -1e8) {
        wrong_right = -right_dynamic_loss_unsafe(next_state, deprel, ref_heads, ref_deprels);
        rewards.push_back(wrong_right);
      } else {
        rewards.push_back(wrong_right);
      }
    } else {
      rewards.push_back(-reduce_dynamic_loss_unsafe(next_state, ref_heads, ref_deprels));
    }
  }
}

void ArcEager::perform_action(State& state, const unsigned& action) {
  if (is_shift(action)) {
    shift_unsafe(state);
  } else if (is_left(action)) {
    left_unsafe(state, parse_label(action));
  } else if (is_right(action)) {
    right_unsafe(state, parse_label(action));
  } else {
    reduce_unsafe(state);
  }
}

bool ArcEager::is_valid_action(const State & state, const unsigned & act) const {
  return false;
}

void ArcEager::get_valid_actions(const State& state, std::vector<unsigned>& valid_actions) {
  BOOST_ASSERT_MSG(false, "Unimplemented.");
  valid_actions.clear();
  unsigned root_id = state.heads.size() - 1;
  unsigned b = state.buffer.back();

  // count the empty heads
  unsigned n_empty_heads = 0;
  for (unsigned i = 1; i < state.stack.size(); ++i) {
    unsigned s = state.stack[i];
    if (state.heads[s] == Corpus::BAD_HED) { n_empty_heads++; }
  }

  if (b < root_id - 1 ||
    (b == root_id - 1 && n_empty_heads == 0) ||
    (b == root_id && state.stack.size() == 1)) {
    valid_actions.push_back(get_shift_id());
  }

  if (state.stack.size() > 1) {
    unsigned s = state.stack.back();
    if (state.heads[s] != Corpus::BAD_HED) {
      valid_actions.push_back(get_reduce_id());
    } else {
      // try LeftArc
      if (b < root_id) {
        for (unsigned i = 0; i < deprel_map.size(); ++i) {
          unsigned act = get_left_id(i);
        }
      } else {
      }     
    }

    // try RightArc
    if (b < root_id - 1 || (b == root_id - 1 && n_empty_heads == 1)) {
      for (unsigned i = 0; i < deprel_map.size(); ++i) {
        unsigned act = get_right_id(i);
      }
    }
  }
  BOOST_ASSERT_MSG(valid_actions.size() > 0, "There should be one or more valid action.");
}

unsigned ArcEager::parse_label(const unsigned& action) {
  BOOST_ASSERT_MSG(action > 1, "SHIFT/REDUCE do not have label.");
  return (action % 2 == 0 ? (action - 2) / 2 : (action - 3) / 2);
}

void ArcEager::get_oracle_actions(const std::vector<unsigned>& ref_heads,
                                  const std::vector<unsigned>& ref_deprels,
                                  std::vector<unsigned>& actions) {
  const unsigned len = ref_heads.size();
  std::vector<unsigned> sigma;
  std::vector<unsigned> heads(len, Corpus::BAD_HED);
  unsigned beta = 0;

  while (!(sigma.size() == 1 && beta == len)) {
    get_oracle_actions_onestep(ref_heads, ref_deprels, sigma, heads, beta, len, actions);
  }
}

unsigned ArcEager::get_structure_action(const unsigned & action) {
  return (action < 3 ? action : (action % 2 == 1 ? 3 : 4));
}

void ArcEager::get_oracle_actions_onestep(const std::vector<unsigned>& ref_heads,
                                          const std::vector<unsigned>& ref_deprels,
                                          std::vector<unsigned>& sigma,
                                          std::vector<unsigned>& heads,
                                          unsigned& beta,
                                          const unsigned& len,
                                          std::vector<unsigned>& actions) {
  if (beta == len) {
    actions.push_back(get_reduce_id());
    sigma.pop_back();
    return;
  }

  if (sigma.size() > 0) {
    unsigned top = sigma.back();
    while (heads[top] != Corpus::BAD_HED) { top = heads[top]; }

    if (ref_heads[top] == beta) {
      if (top == sigma.back()) {
        actions.push_back(get_left_id(ref_deprels[top]));
        heads[top] = beta;
      } else {
        actions.push_back(get_reduce_id());
      }
      sigma.pop_back();
      return;
    }
  }

  if (ref_heads[beta] == Corpus::BAD_HED || ref_heads[beta] > beta) {
    sigma.push_back(beta);
    beta += 1;
    actions.push_back(get_shift_id());
    return;
  } else {
    unsigned top = sigma.back();
    if (ref_heads[beta] == top) {
      actions.push_back(get_right_id(ref_deprels[beta]));
      sigma.push_back(beta);
      heads[beta] = top;
      beta += 1;
    } else {
      actions.push_back(get_reduce_id());
      sigma.pop_back();
    }
  }
}
