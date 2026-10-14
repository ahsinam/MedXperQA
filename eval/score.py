import json, sys
model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-oss-120b"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "groq"
path = f"outputs/{out_dir}/{model}/medxpertqa/zero_shot/cot/medxpertqa_text_output.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]
def report(name, subset):
    if subset:
        c = sum(r["correct"] for r in subset)
        print(f"{name:14s} {c:5d} / {len(subset):5d} correct = {c/len(subset):.2%}")
print(f"Model: {model}")
report("All", rows)
for t in sorted({r["question_type"] for r in rows}):
    report(t, [r for r in rows if r["question_type"] == t])
