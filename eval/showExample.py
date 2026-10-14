import json, textwrap

path = "outputs/groq/openai/gpt-oss-120b/medxpertqa/zero_shot/cot/medxpertqa_text_output.jsonl"
for line in open(path):
    r = json.loads(line)
    if r["id"] == "Text-6":
        question, choices = r["question"].split("Answer Choices:")
        print(f"ID: {r['id']}   ({r['question_type']} · {r['medical_task']})\n")
        print("QUESTION:")
        print(textwrap.fill(question.strip(), 80), "\n")
        print("OPTIONS:")
        for opt in choices.strip().split(" (")[0:]:
            print("  (" + opt.lstrip("(").strip())
        print("\nROUND 1 reply (GPT-OSS 120B):", r["prediction_rationale"])
        print("ROUND 2 reply:", r["response"])
        print("\nPARSED:", r["prediction"][0], "   CORRECT ANSWER:", r["label"][0],
              "   →", "✓ correct" if r["correct"] else "✗ wrong")