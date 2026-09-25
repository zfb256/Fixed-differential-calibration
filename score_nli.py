"""Score ordinary, cautious and premise-removed NLI inputs with a local model."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

RULE = "Classify the relation between the premise and the hypothesis. Treat the premise as true. "
CAUTION = ("Check whether the hypothesis is actually supported by the premise, rather than merely plausible. "
           "Do not assume unstated outcomes, intentions, causes, or details. "
           "Missing evidence is not a contradiction. Preserve entailments that the premise genuinely supports. ")
CONDITIONS = ("base", "cautious", "null_base", "null_cautious")


def answer_tokens(tokenizer):
    # Use bare answer tokens immediately after the assistant newline.
    vocab = tokenizer.get_vocab()
    ids = [vocab[x] for x in "ABC"]
    assert all(tokenizer.decode([i]) == x for i, x in zip(ids, "ABC"))
    return ids


def semantic_logp(lp, order):
    assert sorted(order)==list(range(len(lp)))
    return [float(lp[order.index(label)]) for label in range(len(lp))]


def chat_ids(tokenizer, row, condition, model_type):
    messages=[{'role':'user','content':prompt(row,condition)}]
    if model_type=='glm':
        # GLM's template ends at the assistant role marker, before its newline.
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        assert text.endswith('<|assistant|>')
        return tokenizer.encode(text+'\n',add_special_tokens=False)
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)


def prompt(row, condition):
    order = row.get('answer_order', list(range(row['classes'])))
    assert sorted(order) == list(range(row['classes']))
    alternate = row.get('prompt_variant') == 'paraphrase'
    meanings = ['entailment (the hypothesis must be true).', 'neutral (the hypothesis may be true or false).',
                'contradiction (the hypothesis must be false).'] if row['classes']==3 else [
                'entailment (the hypothesis must be true).', 'not entailment (the hypothesis is not guaranteed by the premise).']
    options = '\n'.join(f'{letter}: {meanings[label]}' for letter,label in zip('ABC',order))
    caution = ("Use only what follows from the premise. Do not fill in missing events, goals, causes, or facts. "
               "An unsupported claim is not necessarily false. Accept conclusions that are logically required by the premise. ") if alternate else CAUTION
    premise = "[No premise is provided.]" if condition.startswith("null_") else row["premise"]
    hypothesis = "[No hypothesis is provided.]" if condition == "premise_only" else row["hypothesis"]
    return (RULE + (caution if "cautious" in condition else "") + "\n" + options +
            "\n\nPremise:\n" + premise + "\n\nHypothesis:\n" + hypothesis +
            "\n\nReply with only the answer letter.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int)
    ap.add_argument('--trust-local-code',action='store_true')
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS)+["premise_only"])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable; run in host GPU environment"
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config=AutoConfig.from_pretrained(args.model,local_files_only=True,trust_remote_code=args.trust_local_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left",trust_remote_code=args.trust_local_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = answer_tokens(tokenizer)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    meta = {"model": args.model, "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256((RULE+CAUTION).encode()).hexdigest(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "max_tokens": args.max_tokens, "batch_size": args.batch_size,
            "answer_token_ids": token_ids, "dtype": "bfloat16", "transformers": __import__("transformers").__version__,
            'model_type':config.model_type,'trust_local_code':args.trust_local_code,
            'assistant_boundary':'append newline after GLM assistant marker' if config.model_type=='glm' else 'native template',
            "torch": torch.__version__, "label_order": ["entailment", "neutral", "contradiction"],
            "binary_nli_labels": ["entailment", "not_entailment"],
            "task_label_orders": {r['dataset']:r['label_names'] for r in rows if 'label_names' in r},
            "conditions": args.conditions, "evaluation": "exploratory", "gpu": torch.cuda.get_device_name(0)}
    meta_path = output.with_suffix(".meta.json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("model", "data_sha256", "prompt_sha256", "code_sha256", "max_tokens", "batch_size", "answer_token_ids", "conditions", "dtype", "transformers", "torch"):
            assert old[key] == meta[key], f"Resume configuration differs: {key}"
    else:
        meta_path.write_text(json.dumps(meta, indent=2))
    done = {}
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            key = (r["id"], r["condition"])
            assert key not in done, key
            done[key] = r
    jobs = []
    for row in rows:
        for condition in args.conditions:
            if (row["id"], condition) in done:
                continue
            ids = chat_ids(tokenizer,row,condition,config.model_type)
            assert len(ids) <= args.max_tokens, f"Overlength input {row['id']}: {len(ids)}; do not silently truncate"
            jobs.append((row, condition, ids))
    jobs.sort(key=lambda x: len(x[2]))
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",trust_remote_code=args.trust_local_code)
    model.eval()
    print(json.dumps({"event": "loaded", "jobs": len(jobs), "rows": len(rows), "gpu": meta["gpu"]}), flush=True)
    started = time.time()
    with output.open("a", buffering=1) as out, torch.inference_mode():
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start+args.batch_size]
            inputs = tokenizer.pad({"input_ids": [x[2] for x in batch]}, padding=True, return_tensors="pt").to("cuda")
            t = time.time()
            # Standard model forward avoids bf16 GEMM-path changes near answer ties.
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            all_lp = logits.log_softmax(-1)
            selected = all_lp[:, token_ids].cpu()
            if start == 0:
                top = logits[0].topk(5)
                print(json.dumps({"first_prompt_top_tokens": [tokenizer.decode([i]) for i in top.indices.tolist()],
                                  "answer_token_ids": token_ids,
                                  "first_prompt_answer_mass": selected[0].logsumexp(-1).exp().item()}), flush=True)
                assert selected.logsumexp(-1).exp().median().item() > 0.01, "Answer tokens have negligible mass; audit tokenization before proceeding"
            torch.cuda.synchronize()
            seconds = time.time()-t
            for i, (row, condition, ids) in enumerate(batch):
                lp = selected[i, :row["classes"]]
                mass = lp.logsumexp(-1).exp().item()
                lp = lp-lp.logsumexp(-1)
                record = {"id": row["id"], "dataset": row["dataset"], "group": row["group"],
                          "label": row.get("label"), "classes": row["classes"], "condition": condition,
                          "logp": semantic_logp(lp,row.get('answer_order',list(range(row['classes'])))),
                          "answer_order": row.get('answer_order',list(range(row['classes']))),
                          "prompt_variant": row.get('prompt_variant','original'),
                          "answer_mass": mass, "tokens": len(ids),
                          "batch_seconds_per_prompt": seconds/len(batch)}
                out.write(json.dumps(record)+"\n")
            if start % (args.batch_size*25) == 0 or start+len(batch) == len(jobs):
                print(json.dumps({"done": start+len(batch), "total": len(jobs),
                    "seconds": round(time.time()-started, 1),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated()/2**30, 2)}), flush=True)
    print("COMPLETE", output, flush=True)
    output.with_suffix(".runtime.json").write_text(json.dumps({"scoring_seconds": time.time()-started,
        "prompts_this_run": len(jobs), "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30}, indent=2))


if __name__ == "__main__":
    main()
