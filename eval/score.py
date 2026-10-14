
import json, sys
# Usage: python score.py MODEL OUTPUT_DIR [DATASET]
#   e.g. python score.py openai/gpt-oss-120b groq
#        python score.py Qwen/Qwen2.5-3B-Instruct hf medxpertqa_sampled
model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "groq"
dataset = sys.argv[3] if len(sys.argv) > 3 else "medxpertqa"
path = f"outputs/{out_dir}/{model}/{dataset}/zero_shot/cot/{dataset}_text_output.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
def report(name, subset):
    if subset:
        c = sum(r["correct"] for r in subset)
        print(f"{name:14s} {c:5d} / {len(subset):5d} correct = {c/len(subset):.2%}")
print(f"Model: {model}   Dataset: {dataset}")
report("All", rows)
for t in sorted({r["question_type"] for r in rows}):
    report(t, [r for r in rows if r["question_type"] == t])
