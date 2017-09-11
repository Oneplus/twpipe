#!/usr/bin/env python
from __future__ import print_function
import sys


def main():
    dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
    n_recall, n_pred, n_gold = 0, 0, 0
    for data in dataset:
        lines = data.splitlines()
        text = None
        gold = set()
        i = 0
        for line in lines:
            if line.startswith('# text = '):
                text = line[len('# text = '):]
            elif not line.startswith('#'):
                word = line.split()[1]
                gold.add((i, len(word)))
                i += len(word)
        assert text is not None, data
        pred = set()
        i = 0
        for word in text.strip().split():
            pred.add((i, len(word)))
            i += len(word)
        for sep in pred:
            if sep in gold:
                n_recall += 1
        n_pred += len(pred)
        n_gold += len(gold)
    f = 2 * float(n_recall) / (n_pred + n_gold)
    print(f)

if __name__ == "__main__":
    main()
