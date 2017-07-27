#ifndef TREE_H
#define TREE_H

#include <vector>
#include "corpus.h"

struct DependencyUtils {
  typedef std::vector<unsigned> node_t;
  typedef std::vector<node_t> tree_t;

  static bool is_tree(const std::vector<unsigned>& heads) {
    tree_t tree(heads.size());
    unsigned root = Corpus::BAD_HED;

    for (unsigned modifier = 0; modifier < heads.size(); ++ modifier) {
      unsigned head = heads[modifier];
      if (head == Corpus::BAD_HED) {
        root = modifier;
      } else if (head == Corpus::REMOVED_HED) {
        continue;
      } else if (head >= heads.size()) {
        return false;
      } else {
        tree[head].push_back(modifier);
      }
    }
    std::vector<bool> visited(heads.size(), false);
    if (!is_tree_travel(root, tree, visited)) {
      return false;
    }
    for (unsigned modifier = 0; modifier < heads.size(); ++modifier) {
      unsigned head = heads[modifier];
      if (head == Corpus::REMOVED_HED) { continue; }
      if (!visited[modifier]) { return false; }
    }
    return true;
  }

  static bool is_tree_travel(int now, const tree_t& tree, std::vector<bool>& visited) {
    if (visited[now]) { return false; }

    visited[now] = true;
    for (unsigned next: tree[now]) {
      if (!is_tree_travel(next, tree, visited)) { return false; }
    }
    return true;
  }

  static bool is_non_projective(const std::vector<unsigned>& heads) {
    for (unsigned modifier = 0; modifier < heads.size(); ++ modifier) {
      unsigned head = heads[modifier];
      if (head == Corpus::REMOVED_HED) { continue; }

      if (head < modifier) {
        for (unsigned from = head + 1; from < modifier; ++ from) {
          unsigned to = heads[from];
          if (to == Corpus::REMOVED_HED) { continue; }
          if (to < head || to > modifier) { return true; }
        }
      } else {
        for (unsigned from = modifier + 1; from < head && from < heads.size(); ++ from) {
          unsigned to = heads[from];
          if (to == Corpus::REMOVED_HED) { continue; }
          if (to < modifier || to > head) { return true; }
        }
      }
    }
    return false;
  }

  static bool is_projective(const std::vector<unsigned>& heads) {
    return !is_non_projective(heads);
  }

  static bool is_non_projective(const ParseUnits& parse_units) {
    std::vector<unsigned> heads;
    std::vector<unsigned> deprels;
    parse_to_vector(parse_units, heads, deprels);
    return is_non_projective(heads);
  }

  static bool is_projective(const ParseUnits& parse_units) {
    std::vector<unsigned> heads;
    std::vector<unsigned> deprels;
    parse_to_vector(parse_units, heads, deprels);
    return is_projective(heads);
  }

  static bool is_tree(const ParseUnits& parse_units) {
    std::vector<unsigned> heads;
    std::vector<unsigned> deprels;
    parse_to_vector(parse_units, heads, deprels);
    return is_tree(heads);
  }

  static bool is_tree_and_projective(const ParseUnits& parse_units) {
    std::vector<unsigned> heads;
    std::vector<unsigned> deprels;
    parse_to_vector(parse_units, heads, deprels);
    return (is_tree(heads) && is_projective(heads));
  }
};

#endif  //  end for TREE_H
