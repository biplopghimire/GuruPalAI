import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)


# model_id = "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

teacher_notes = """
Vocabulary List:
1. Benevolent – kind and generous
2. Cautious – careful to avoid danger or mistakes
3. Ambitious – having a strong desire to succeed
"""

student_question = "What does benevolent mean in context?"

prompt = f"""You are a helpful AI teacher. Use the information below to answer the student's question.

Notes:
{teacher_notes}

Student Question: {student_question}
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# Decode and print answer
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer_only = full_output[len(prompt):].strip()
print("\n🧠 Answer:\n", answer_only)