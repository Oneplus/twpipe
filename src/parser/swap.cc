#include "swap.h"
#include "twpipe/logging.h"
#include "twpipe/corpus.h"
#include "twpipe/alphabet_collection.h"

namespace twpipe {

Swap::Swap() : TransitionSystem() {
  Alphabet & map = AlphabetCollection::get()->deprel_map;
  n_actions = 2 * map.size() + 2;

  action_names.push_back("SHIFT");  // 0
  action_names.push_back("SWAP");   // 1
  for (unsigned i = 0; i < map.size(); ++i) {
    action_names.push_back("LEFT-" + map.get(i));
    action_names.push_back("RIGHT-" + map.get(i));
  }
  _INFO << "TransitionSystem:: show action names:";
  for (const auto& action_name : action_names) {
    _INFO << "- " << action_name;
  }
}

std::string Swap::name() const {
  return "swap";
}

std::string Swap::name(unsigned id) const {
  BOOST_ASSERT_MSG(id < action_names.size(), "id in illegal range");
  return action_names[id];
}

bool Swap::allow_nonprojective() const {
  return true;
}

unsigned Swap::num_actions() const { return n_actions; }

void Swap::get_transition_costs(const State & state,
                                const std::vector<unsigned>& actions,
                                const std::vector<unsigned>& ref_heads,
                                const std::vector<unsigned>& ref_deprels,
                                std::vector<float>& costs) {
  BOOST_ASSERT_MSG(false, "Inefficient to define dynamic oracle for SWAP system");
}

unsigned Swap::get_structure_action(const unsigned & action) {
  return (action < 3 ? action : (action % 2 == 1 ? 3 : 4));
}

void Swap::perform_action(State & state, const unsigned & action) {
  if (is_shift(action)) {
    shift_unsafe(state);
  } else if (is_swap(action)) {
    swap_unsafe(state);
  } else if (is_left(action)) {
    left_unsafe(state, parse_label(action));
  } else {
    right_unsafe(state, parse_label(action));
  }
}

void Swap::get_oracle_actions(const std::vector<unsigned>& ref_heads,
                              const std::vector<unsigned>& ref_deprels,
                              std::vector<unsigned>& actions) {
  unsigned N = ref_heads.size();
  unsigned root = Corpus::BAD_HED;
  std::vector<std::vector<unsigned>> tree(N);
  for (unsigned i = 0; i < N; ++i) {
    unsigned ref_head = ref_heads[i];
    if (ref_head == Corpus::BAD_HED) {
      BOOST_ASSERT_MSG(root == Corpus::BAD_HED, "There should be only one root.");
      root = i;
    } else {
      tree[ref_head].push_back(i);
    }
  }

  unsigned timestamp = 0;
  std::vector<unsigned> orders(N, Corpus::BAD_HED);
  get_oracle_actions_calculate_orders(root, tree, orders, timestamp);

  std::vector<unsigned> mpc(N, 0);
  get_oracle_actions_calculate_mpc(root, tree, mpc);

  std::vector<unsigned> sigma;
  std::vector<unsigned> beta;
  std::vector<unsigned> heads(N, Corpus::BAD_HED);
  for (int i = N - 1; i >= 0; --i) { beta.push_back(i); }

  while (!(sigma.size() == 1 && beta.empty())) {
    get_oracle_actions_onestep_improved(ref_heads, ref_deprels,
                                        tree, orders, mpc,
                                        sigma, beta, heads,
                                        actions);
  }
}

void Swap::get_oracle_actions_calculate_orders(const unsigned & root,
                                               const std::vector<std::vector<unsigned>>& tree,
                                               std::vector<unsigned>& orders,
                                               unsigned & timestamp) {
  const std::vector<unsigned>& children = tree[root];
  if (children.size() == 0) {
    orders[root] = timestamp;
    timestamp += 1;
    return;
  }

  unsigned i;
  for (i = 0; i < children.size() && children[i] < root; ++i) {
    int child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }

  orders[root] = timestamp;
  timestamp += 1;

  for (; i < children.size(); ++i) {
    unsigned child = children[i];
    get_oracle_actions_calculate_orders(child, tree, orders, timestamp);
  }
}

Swap::mpc_result_t Swap::get_oracle_actions_calculate_mpc(const unsigned & root,
                                                          const std::vector<std::vector<unsigned>>& tree,
                                                          std::vector<unsigned>& mpc) {
  const std::vector<unsigned>& children = tree[root];
  if (children.size() == 0) {
    mpc[root] = root;
    return std::make_tuple(true, root, root);
  }

  unsigned left = root, right = root;
  bool overall = true;

  int pivot = -1;
  for (pivot = 0; pivot < children.size() && children[pivot] < root; ++pivot);

  for (int i = pivot - 1; i >= 0; --i) {
    unsigned child = children[i];
    mpc_result_t result = get_oracle_actions_calculate_mpc(child, tree, mpc);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && std::get<2>(result) + 1 == left) {
      left = std::get<1>(result);
    } else {
      overall = false;
    }
  }

  for (int i = pivot; i < children.size(); ++i) {
    unsigned child = children[i];
    mpc_result_t result = get_oracle_actions_calculate_mpc(child, tree, mpc);
    overall = overall && std::get<0>(result);
    if (std::get<0>(result) == true && right + 1 == std::get<1>(result)) {
      right = std::get<2>(result);
    } else {
      overall = false;
    }
  }

  for (int i = left; i <= right; ++i) { mpc[i] = root; }

  return std::make_tuple(overall, left, right);
}

void Swap::get_oracle_actions_onestep(const std::vector<unsigned>& ref_heads,
                                      const std::vector<unsigned>& ref_deprels,
                                      const std::vector<std::vector<unsigned>>& tree,
                                      const std::vector<unsigned>& orders,
                                      std::vector<unsigned>& sigma,
                                      std::vector<unsigned>& beta,
                                      std::vector<unsigned>& heads,
                                      std::vector<unsigned>& actions) {
  if (sigma.size() < 2) {
    actions.push_back(get_shift_id());
    sigma.push_back(beta.back());
    beta.pop_back();
    return;
  }

  unsigned top0 = sigma.back();
  unsigned top1 = sigma[sigma.size() - 2];

  if (ref_heads[top1] == top0) {
    bool all_found = true;
    for (unsigned c : tree[top1]) {
      if (heads[c] == Corpus::BAD_HED) { all_found = false; }
    }
    if (all_found) {
      actions.push_back(get_left_id(ref_deprels[top1]));
      sigma.pop_back();
      sigma.back() = top0;
      heads[top1] = top0;
      return;
    }
  }

  if (ref_heads[top0] == top1) {
    bool all_found = true;
    for (unsigned c : tree[top0]) {
      if (heads[c] == Corpus::BAD_HED) { all_found = false; }
    }
    if (all_found) {
      actions.push_back(get_right_id(ref_deprels[top0]));
      sigma.pop_back();
      heads[top0] = top1;
      return;
    }
  }

  if (orders[top0] < orders[top1]) {
    actions.push_back(get_swap_id());
    sigma.pop_back();
    sigma.back() = top0;
    beta.push_back(top1);
  } else {
    actions.push_back(get_shift_id());
    sigma.push_back(beta.back());
    beta.pop_back();
  }
}

void Swap::get_oracle_actions_onestep_improved(const std::vector<unsigned>& ref_heads,
                                               const std::vector<unsigned>& ref_deprels,
                                               const std::vector<std::vector<unsigned>>& tree,
                                               const std::vector<unsigned>& orders,
                                               const std::vector<unsigned>& mpc,
                                               std::vector<unsigned>& sigma,
                                               std::vector<unsigned>& beta,
                                               std::vector<unsigned>& heads,
                                               std::vector<unsigned>& actions) {
  if (sigma.size() < 2) {
    actions.push_back(get_shift_id());
    sigma.push_back(beta.back());
    beta.pop_back();
    return;
  }

  unsigned top0 = sigma.back();
  unsigned top1 = sigma[sigma.size() - 2];

  if (ref_heads[top1] == top0) {
    bool all_found = true;
    for (unsigned c : tree[top1]) {
      if (heads[c] == Corpus::BAD_HED) { all_found = false; }
    }
    if (all_found) {
      actions.push_back(get_left_id(ref_deprels[top1]));
      sigma.pop_back();
      sigma.back() = top0;
      heads[top1] = top0;
      return;
    }
  }

  if (ref_heads[top0] == top1) {
    bool all_found = true;
    for (unsigned c : tree[top0]) {
      if (heads[c] == Corpus::BAD_HED) { all_found = false; }
    }
    if (all_found) {
      actions.push_back(get_right_id(ref_deprels[top0]));
      sigma.pop_back();
      heads[top0] = top1;
      return;
    }
  }

  unsigned k = beta.empty() ? Corpus::BAD_HED : beta.back();
  if ((orders[top0] < orders[top1]) && (k == Corpus::BAD_HED || mpc[top0] != mpc[k])) {
    actions.push_back(get_swap_id());
    sigma.pop_back();
    sigma.back() = top0;
    beta.push_back(top1);
  } else {
    actions.push_back(get_shift_id());
    sigma.push_back(beta.back());
    beta.pop_back();
  }
}

void Swap::shift_unsafe(State & state) const {
  state.stack.push_back(state.buffer.back());
  state.buffer.pop_back();
}

void Swap::swap_unsafe(State & state) const {
  unsigned j = state.stack.back(); state.stack.pop_back();
  unsigned i = state.stack.back(); state.stack.pop_back();
  state.stack.push_back(j);
  state.buffer.push_back(i);
}

void Swap::left_unsafe(State & state, const unsigned & deprel) const {
  unsigned hed = state.stack.back(); state.stack.pop_back();
  unsigned mod = state.stack.back(); state.stack.back() = hed;
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

void Swap::right_unsafe(State & state, const unsigned & deprel) const {
  unsigned mod = state.stack.back(); state.stack.pop_back();
  unsigned hed = state.stack.back();
  state.heads[mod] = hed;
  state.deprels[mod] = deprel;
}

unsigned Swap::get_shift_id() const { return 0; }
unsigned Swap::get_swap_id() const { return 1; }
unsigned Swap::get_left_id(const unsigned & deprel) const { return deprel * 2 + 2; }
unsigned Swap::get_right_id(const unsigned & deprel) const { return deprel * 2 + 3; }

bool Swap::is_shift(const unsigned & action) { return action == 0; }
bool Swap::is_swap(const unsigned & action) { return action == 1; }
bool Swap::is_left(const unsigned & action) { return action > 1 && action % 2 == 0; }
bool Swap::is_right(const unsigned & action) { return action > 1 && action % 2 == 1; }

unsigned Swap::parse_label(const unsigned & action) const {
  BOOST_ASSERT_MSG(action > 1, "SHIFT ans SWAP do not have label.");
  return (action % 2 == 0 ? (action - 2) / 2 : (action - 3) / 2);
}

void Swap::get_valid_actions(const State & state,
                             std::vector<unsigned>& valid_actions) {
  valid_actions.clear();
  for (unsigned a = 0; a < n_actions; ++a) {
    //if (!is_valid_action(state, action_names[a])) { continue; }
    if (!is_valid_action(state, a)) { continue; }
    valid_actions.push_back(a);
  }
  BOOST_ASSERT_MSG(valid_actions.size() > 0, "There should be one or more valid action.");
}

bool Swap::is_valid_action(const State& state, const unsigned& act) const {
  bool is_shift = (act == 0);
  bool is_swap = (act == 1);
  bool is_reduce = (!is_shift && !is_swap);

  if (is_shift && state.buffer.size() == 1) { return false; }
  if (is_swap && (state.stack.size() < 3 || state.buffer.size() == 1)) { return false; }
  if (is_swap && state.stack[state.stack.size() - 2] > state.stack.back()) { return false; }
  if (is_reduce && state.stack.size() < 3) { return false; }
  if (state.buffer.size() == 1 && !is_left(act)) { return false; }
  return true;
}

}