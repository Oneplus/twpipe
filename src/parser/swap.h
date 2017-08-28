#ifndef __TWPIPE_PARSER_SWAP_H__
#define __TWPIPE_PARSER_SWAP_H__

#include "system.h"

namespace twpipe {

struct Swap : public TransitionSystem {
  typedef std::tuple<bool, unsigned, unsigned> mpc_result_t;

  unsigned n_actions;
  std::vector<std::string> action_names;

  Swap(const Alphabet& deprel_map);

  std::string name(unsigned id) const override;

  bool allow_nonprojective() const override;

  unsigned num_actions() const override;

  unsigned num_deprels() const override;

  void get_transition_costs(const State& state,
                            const std::vector<unsigned>& actions,
                            const std::vector<unsigned>& ref_heads,
                            const std::vector<unsigned>& ref_deprels,
                            std::vector<float>& rewards);

  unsigned get_structure_action(const unsigned & action) override;

  void perform_action(State& state, const unsigned& action) override;

  void get_valid_actions(const State& state,
                         std::vector<unsigned>& valid_actions) override;

  void get_oracle_actions(const std::vector<unsigned>& heads,
                          const std::vector<unsigned>& deprels,
                          std::vector<unsigned>& actions) override;

  bool is_valid_action(const State& state, const unsigned& act) const override;

  void get_oracle_actions_calculate_orders(const unsigned & root,
                                           const std::vector<std::vector<unsigned>>& tree,
                                           std::vector<unsigned>& orders,
                                           unsigned& timestamp);

  mpc_result_t get_oracle_actions_calculate_mpc(const unsigned & root,
                                                const std::vector<std::vector<unsigned>>& tree,
                                                std::vector<unsigned>& mpc);

  void get_oracle_actions_onestep(const std::vector<unsigned>& ref_heads,
                                  const std::vector<unsigned>& ref_deprels,
                                  const std::vector<std::vector<unsigned>>& tree,
                                  const std::vector<unsigned>& orders,
                                  std::vector<unsigned>& sigma,
                                  std::vector<unsigned>& beta,
                                  std::vector<unsigned>& heads,
                                  std::vector<unsigned>& actions);

  void get_oracle_actions_onestep_improved(const std::vector<unsigned>& ref_heads,
                                           const std::vector<unsigned>& ref_deprels,
                                           const std::vector<std::vector<unsigned>>& tree,
                                           const std::vector<unsigned>& orders,
                                           const std::vector<unsigned>& mpc,
                                           std::vector<unsigned>& sigma,
                                           std::vector<unsigned>& beta,
                                           std::vector<unsigned>& heads,
                                           std::vector<unsigned>& actions);

  void shift_unsafe(State& state) const;
  void swap_unsafe(State& state) const;
  void left_unsafe(State& state, const unsigned& deprel) const;
  void right_unsafe(State& state, const unsigned& deprel) const;

  static bool is_shift(const unsigned& action);
  static bool is_left(const unsigned& action);
  static bool is_right(const unsigned& action);
  static bool is_swap(const unsigned& action);

  unsigned get_shift_id() const;
  unsigned get_swap_id() const;
  unsigned get_left_id(const unsigned & deprel) const;
  unsigned get_right_id(const unsigned & deprel) const;

  unsigned parse_label(const unsigned & action) const;
};

}

#endif  //  end for SWAP_H