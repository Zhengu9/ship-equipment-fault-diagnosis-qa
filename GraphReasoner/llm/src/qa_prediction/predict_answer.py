import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import argparse
from tqdm import tqdm
from llms.language_models import get_registed_model
import os
from .datasets import load_dataset
from qa_prediction.evaluate_results import eval_result
import json
from multiprocessing import Pool
from qa_prediction.build_qa_input import PromptBuilder
from functools import partial

import json

with open('entities_names.json') as f:
    entities_names = json.load(f)
names_entities = {v: k for k, v in entities_names.items()}

import re
import string
from difflib import SequenceMatcher
# When True, always use top GNN candidate as final prediction (force GNN answer)
FORCE_GNN = False
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def load_gnn_rag(g_data_file, g_data_file2=None):
    data_file_d = {}
    data_file_gnn = {}

    data_file = os.path.dirname(g_data_file) + "/test.json"
    with open(data_file) as f_in, open(g_data_file) as fg:
        for line, lineg in (zip(f_in, fg)):
            line = json.loads(line)
            lineg = json.loads(lineg)
            
            data_file_d[line["id"]] = line
            data_file_gnn[line["id"]] = lineg
        print("ok1")
    if g_data_file2 is not None:
        data_file = os.path.dirname(g_data_file2) + "/test.json"
        with open(data_file) as f_in, open(g_data_file2) as fg:
            for line, lineg in (zip(f_in, fg)):
                line = json.loads(line)
                lineg = json.loads(lineg)
                
                cand1 = data_file_gnn[line["id"]]["cand"]
                cand2 =  lineg["cand"]

                for c2 in cand2: #c[0] entity c[1] score
                    found=False
                    for c1 in cand1:
                        if c2[0] == c1[0]:
                            if c2[1] > c1[1]: c1[1] = c2[1]
                            found=True
                            break
                    if not found:
                        cand1.append(c2)
                cand1 = sorted(cand1, key=lambda x: x[1], reverse=True)
                data_file_gnn[line["id"]]["cand"] = cand1
            data_file_gnn[line["id"]].update({"cand2": lineg["cand"]})
            print("ok2")

    return data_file_gnn


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w", encoding="utf-8")
        return fout, []
    else:
        with open(path, "r", encoding="utf-8") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a", encoding="utf-8")
        return fout, processed_results


def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def prediction(data, processed_list, input_builder, model, encrypt=False, data_file_gnn=None):
    question = data["question"]
    # support datasets that use either 'answer' (single) or 'answers' (list)
    if "answer" in data:
        answer = data["answer"]
    else:
        answer = ""
        if "answers" in data and isinstance(data["answers"], list) and len(data["answers"])>0:
            # some files store answers as list of dicts like [{'kb_id':..., 'text':...}]
            first = data["answers"][0]
            if isinstance(first, dict) and "text" in first:
                answer = first["text"]
            elif isinstance(first, str):
                answer = first
    entities = []
    if 'q_entity' in data:
        entities = data['q_entity']
    elif 'entities' in data:
        entities = data['entities']
    
    data["cand"] = None
    id = data["id"]
    if data_file_gnn is not None:
        
        lineg = data_file_gnn[data["id"]]
        cand = lineg['cand'] 
        predictiong = []
        for c in cand:
            if c[0] in entities_names:
                predictiong.append(entities_names[c[0]])
            else:
                predictiong.append(c[0])
        data["cand"] = predictiong
    
    
    if id in processed_list:
        return None
    
    if model is None:
        prediction = input_builder.direct_answer(data)
        return {
            "id": id,
            "question": question,
            "prediction": prediction,
            "ground_truth": answer,
            "input": question,
        }
    
    # normalize dataset fields to what PromptBuilder expects
    if 'q_entity' not in data and 'entities' in data:
        # convert list of dicts or strings to list of ids/texts
        new_q = []
        for e in data['entities']:
            if isinstance(e, dict):
                if 'kb_id' in e and e['kb_id'] is not None and str(e['kb_id']).lower() != 'nan':
                    new_q.append(e['kb_id'])
                elif 'text' in e:
                    new_q.append(e['text'])
            elif isinstance(e, str):
                new_q.append(e)
        data['q_entity'] = new_q
    if 'choices' not in data:
        data['choices'] = []
    if 'graph' not in data and 'subgraph' in data:
        # normalize subgraph shape: some datasets store subgraph as dict{'tuples': [...], 'entities': [...]}
        # while utils.build_graph expects an iterable of triplets (h, r, t).
        if isinstance(data['subgraph'], dict) and 'tuples' in data['subgraph']:
            data['graph'] = data['subgraph']['tuples']
        else:
            data['graph'] = data['subgraph']

    input = input_builder.process_input(data)
    raw_prediction = model.generate_sentence(input)
    if raw_prediction is None:
        return None
    raw_prediction = raw_prediction.strip()
    # If user requested to force GNN, always select top candidate when available
    if FORCE_GNN and data.get("cand") and isinstance(data.get("cand"), list) and len(data.get("cand"))>0:
        final_prediction = data["cand"][0]
    else:
        # Enforce using GNN-provided candidates: pick the candidate with highest string-similarity
        # to the LLM raw output. If similarity is low, fallback to the top candidate.
        final_prediction = raw_prediction
    if data.get("cand") and isinstance(data.get("cand"), list) and len(data.get("cand")) > 0:
        best = None
        best_ratio = 0.0
        for c in data["cand"]:
            try:
                ratio = SequenceMatcher(None, raw_prediction, c).ratio()
            except Exception:
                ratio = 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best = c
        # threshold: require at least moderate similarity, otherwise try to pick a coded candidate
        if best is not None and best_ratio >= 0.4:
            final_prediction = best
        else:
            # prefer candidates that look like coded labels (contain PSC or FSC or shortChinese+code)
            code_candidate = None
            for c in data["cand"]:
                try:
                    if re.search(r"PSC|FSC", c):
                        code_candidate = c
                        break
                except Exception:
                    continue
            if code_candidate is not None:
                final_prediction = code_candidate
            else:
                final_prediction = data["cand"][0]

    result = {
        "id": id,
        "question": question,
        "prediction": final_prediction,
        "llm_raw_prediction": raw_prediction,
        "candidates": data.get("cand"),
        "ground_truth": answer,
        "input": input,
    }
    return result


def main(args, LLM):
    global FORCE_GNN
    FORCE_GNN = getattr(args, 'force_gnn', False)
    input_file = os.path.join(args.data_path, args.d)
    rule_postfix = "no_rule"
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = utils.load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        if args.use_true:
            rule_postfix = "ground_rule"
        elif args.use_random:
            rule_postfix = "random_rule"

    data_file_gnn = None
    if os.path.exists(args.rule_path_g1):
        if not os.path.exists(args.rule_path_g2):
            data_file_gnn = load_gnn_rag(args.rule_path_g1)
        else: 
            data_file_gnn = load_gnn_rag(args.rule_path_g1, args.rule_path_g2)



    if args.cot:
        rule_postfix += "_cot"
    if args.explain:
        rule_postfix += "_explain"
    if args.filter_empty:
        rule_postfix += "_filter_empty"
    if args.each_line:
        rule_postfix += "_each_line"
        
    print("Load dataset from finished")
    output_dir = os.path.join(
        args.predict_path, args.d, args.model_name, args.split, rule_postfix, str(args.encrypt)
    )
    print("Save results to: ", output_dir)
    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.prompt_path,
            args.encrypt,
            args.add_rule,
            use_true=args.use_true,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.tokenize,
        )
        print("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(
            args.prompt_path, args.encrypt,args.add_rule, use_true=args.use_true
        )

    # Save args file (UTF-8)
    with open(os.path.join(output_dir, "args.txt"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2, ensure_ascii=False)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    if args.n > 1:
        with Pool(args.n) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                        encrypt=args.encrypt,
                        data_file_gnn=data_file_gnn

                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res, ensure_ascii=False))
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction(data, processed_list, input_builder, model, encrypt=args.encrypt, data_file_gnn=data_file_gnn)
            if res is not None:
                if args.debug:
                    print(json.dumps(res, ensure_ascii=False))
                fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                fout.flush()
    fout.close()

    # Only run evaluation if not explicitly skipped
    if not getattr(args, 'skip_eval', False):
        eval_result(output_file, encrypt=args.encrypt)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    argparser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")
    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="gpt-3.5-turbo",
    )
    argparser.add_argument(
        "--prompt_path",
        type=str,
        help="prompt_path",
        default="prompts/llama2_predict.txt",
    )
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp/RoG/test/predictions_3_False.jsonl",
    )
    argparser.add_argument(
        "--rule_path_g1",
        type=str,
        default="results/gnn/webqsp/RoG/test/rearev-sbert/test.info",
    )
    argparser.add_argument(
        "--rule_path_g2",
        type=str,
        default=None,
    )
    argparser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    argparser.add_argument("--force_gnn", action="store_true", help="force using top GNN candidate as final prediction")
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")

    argparser.add_argument("--encrypt", action="store_true")
    argparser.add_argument("--skip_eval", action="store_true", help="skip evaluation and only write predictions")

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()

    main(args, LLM)
