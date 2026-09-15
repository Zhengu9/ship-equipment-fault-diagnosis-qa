#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_GraphReasoner_main(args_list, timeout=None):
    try:
        proc = subprocess.run(args_list, text=True, capture_output=True, timeout=timeout)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"GraphReasoner.main exited with {proc.returncode}")
        return True
    except subprocess.TimeoutExpired as e:
        print(f"GraphReasoner.main timed out: {e}", file=sys.stderr)
        return False


def load_info_file(path):
    objs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                continue
    return objs


def best_matches(objs, query, topk=5):
    sims = []
    for o in objs:
        q = o.get('question', '')
        try:
            r = SequenceMatcher(None, query, q).ratio()
        except Exception:
            r = 0.0
        sims.append(r)
    idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:topk]
    return idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default=str(Path(__file__).resolve().parents[1] / 'data' / 'ship_cwq'))
    parser.add_argument('--lm', default='xlm')
    parser.add_argument('--checkpoint_dir', default=str(Path(__file__).resolve().parents[1] / 'checkpoint' / 'ship_cwq'))
    parser.add_argument('--experiment_name', default='ship_cwq')
    parser.add_argument('--load_experiment', default=None, help='checkpoint filename under checkpoint_dir')
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_iter', type=int, default=2)
    parser.add_argument('--num_ins', type=int, default=3)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--relation_word_emb', action='store_true')
    parser.add_argument('--is_eval', action='store_true')
    parser.add_argument('--info_file', default=str(Path(__file__).resolve().parents[1] / 'checkpoint' / 'ship_cwq' / 'ship_cwq_test.info'), help='path to existing _test.info file (skip running GraphReasoner.main)')
    parser.add_argument('--question', default=None, help='user question to match against info file')
    parser.add_argument('--topk', type=int, default=5, help='top-k matched questions to collect')
    parser.add_argument('--timeout', type=int, default=3600, help='timeout for running GraphReasoner.main')
    args = parser.parse_args()

    info_path = None
    if args.info_file:
        info_path = Path(args.info_file)
        if not info_path.exists():
            print(f"info file not found: {info_path}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.load_experiment is None:
            print("--load_experiment is required when not providing --info_file", file=sys.stderr)
            sys.exit(1)
        cmd = [
            sys.executable, '-m', 'GraphReasoner.main', 'ReaRev',
            '--data_folder', args.data_folder,
            '--lm', args.lm,
            '--checkpoint_dir', args.checkpoint_dir,
            '--experiment_name', args.experiment_name,
            '--test_batch_size', str(args.test_batch_size),
            '--num_iter', str(args.num_iter),
            '--num_ins', str(args.num_ins),
            '--num_epoch', str(args.num_epoch),
        ]
        if args.relation_word_emb:
            cmd.append('--relation_word_emb')
        cmd.append('--is_eval')
        cmd.extend(['--load_experiment', args.load_experiment])

        print('Running GraphReasoner inference via:', ' '.join(cmd), file=sys.stderr)
        ok = run_GraphReasoner_main(cmd, timeout=args.timeout)
        if not ok:
            print('GraphReasoner run failed or timed out', file=sys.stderr)
            sys.exit(1)

        info_path = Path(args.checkpoint_dir) / f"{args.experiment_name}_test.info"
        if not info_path.exists():
            print(f"Expected info file not found: {info_path}", file=sys.stderr)
            sys.exit(1)

    objs = load_info_file(str(info_path))
    if args.question is None:
        print(json.dumps({
            'info_file': str(info_path),
            'num_cases': len(objs)
        }, ensure_ascii=False))
        return

    idxs = best_matches(objs, args.question, topk=args.topk)
    merged = []
    seen = set()
    for i in idxs:
        o = objs[i]
        answers = o.get('answers', [])
        if isinstance(answers, list):
            for a in answers:
                if a and a not in seen:
                    seen.add(a)
                    merged.append(a)
        else:
            a = answers
            if a and a not in seen:
                seen.add(a)
                merged.append(a)

    entities_path = Path(__file__).resolve().parents[1] / 'entities_names.json'
    entities_map = {}
    try:
        with open(entities_path, 'r', encoding='utf-8') as f:
            entities_map = json.load(f)
    except Exception:
        try:
            with open(str(REPO_ROOT / 'GraphReasoner' / 'entities_names.json'), 'r', encoding='utf-8') as f:
                entities_map = json.load(f)
        except Exception:
            entities_map = {}

    mapped_candidates = []
    for a in merged:
        orig = str(a)
        mapped = None
        if orig in entities_map:
            mapped = entities_map[orig]
        else:
            num = None
            mnum = re.match(r"^(\d+)\.?$", orig)
            if mnum:
                num = mnum.group(1)
            if num is not None:
                for k, v in entities_map.items():
                    if v.strip().startswith(f"{num}.") or v.strip().startswith(f"{num} "):
                        mapped = v
                        break
            if mapped is None:
                for k, v in entities_map.items():
                    if v == orig:
                        mapped = v
                        break
        if mapped is None:
            mapped = orig
        mapped_candidates.append(mapped)

    seen_texts = set()
    unique_candidates = []
    for txt in mapped_candidates:
        t = txt.strip() if isinstance(txt, str) else str(txt)
        if not t:
            continue
        if t not in seen_texts:
            seen_texts.add(t)
            unique_candidates.append({'text': t, 'score': 1.0, 'source': 'GraphReasoner'})

    print(json.dumps({'candidates': unique_candidates}, ensure_ascii=False))


if __name__ == '__main__':
    main()
