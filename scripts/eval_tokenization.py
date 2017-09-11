#!/usr/bin/env python
from __future__ import print_function
import sys
import argparse
import difflib


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('--system', help='the path to the system output file.')
    cmd.add_argument('--answer', help='the path to the gold standard file.')
    cmd.add_argument('--detail', action='store_true', default=False, help='show the detail.')
    args = cmd.parse_args()
    answers = {}
    for data in open(args.answer, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        words = [line.split()[1] for line in lines if not line.startswith('#')]
        key = ''.join(words)
        answers[key] = {"data": data, "segmentation": set()}
        offset = 0
        for word in words:
            answers[key]["segmentation"].add((offset, offset + len(word)))
            offset += len(word)

    n_pred, n_gold, n_recall = 0, 0, 0
    for data in open(args.system, 'r').read().strip().split('\n\n'):
        lines = data.splitlines()
        words = [line.split()[1] for line in lines if not line.startswith('#')]
        key = ''.join(words)
        if key not in answers:
            print('key {0} not found in answer'.format(key), file=sys.stderr)
            continue
        answer = answers[key]
        segmentation = answer["segmentation"]
        offset, n_corr = 0, 0
        for word in words:
            if (offset, offset + len(word)) in segmentation:
                n_corr += 1
            offset += len(word)
        n_recall += n_corr
        if args.detail and n_corr != len(segmentation):
            answer_words = [line.split()[1] for line in answer["data"].splitlines() if not line.startswith('#')]
            for line in difflib.context_diff(answer_words, words, fromfile='answer', tofile='system'):
                print(line, file=sys.stderr)
        n_pred += len(words)
        n_gold += len(segmentation)

    p = float(n_recall) / n_pred
    r = float(n_recall) / n_gold
    f = 2 * p * r / (p + r)
    print('p:{0}, r:{1}, f:{2}'.format(p, r, f))


if __name__ == "__main__":
    main()
