import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


class CodeCommentDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

code_snippets = ["""def add(a, b):
    return a + b""",
                 """def problema1(text: str) -> str:
        last_word = ""
        for word in text.split(" "):
            if last_word < word:
                last_word = word

        return last_word""",
                 """def subtract(a, b):
                    return a - b"""]
comments = ["This function adds two numbers",
            "this function will determine the first word in alphabetical order.",
            "this function subtracts the second number from the first"]

code_snippets.extend([
    """def multiply(a, b):
        return a * b""",
    """def divide(a, b):
        if b != 0:
            return a / b
        else:
            return 'Error: Division by zero'""",
    """def power(a, b):
        return a ** b""",
    """def sqrt(a):
        if a >= 0:
            return a ** 0.5
        else:
            return 'Error: Negative input'""",
    """def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)""",
    """def sum_list_integers(int_list):
    return sum(int_list)""",
    """def sum_floats(float_list):
    return sum(float_list)""",
    """def add_matrices(matrix1, matrix2):
    result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result

# Example matrices
matrix_a = [[1, 2, 3], [4, 5, 6]]
matrix_b = [[7, 8, 9], [10, 11, 12]]

# Adding the matrices
result_matrix = add_matrices(matrix_a, matrix_b)
print(result_matrix)
"""
])

comments.extend([
    "This function multiplies two numbers",
    "This function divides the first number by the second, but checks if the second number is zero to avoid division "
    "by zero error",
    "This function raises the first number to the power of the second number",
    "This function returns the square root of a number, but checks if the number is negative to avoid math error",
    "This function calculates the factorial of a number using recursion",
    "This is a method that sums a list of ints",
    "This is a method that sums a list of float",
    "This is a function that adds two matrices given as parameters"
])

tokenized_inputs = tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True)
tokenized_outputs = tokenizer(comments, return_tensors="pt", padding=True, truncation=True)

train_dataset = CodeCommentDataset(tokenized_inputs)
eval_dataset = CodeCommentDataset(tokenized_outputs)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./results-2',
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,

)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
model.save_pretrained('fine-tuned-gpt2-2')
tokenizer.save_pretrained('fine-tuned-gpt2-2')


def generate_comment(code_snippet):
    tokenizer = GPT2Tokenizer.from_pretrained("fine-tuned-gpt2-2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("fine-tuned-gpt2-2")
    model.eval()

    input_ids = tokenizer.encode(code_snippet, return_tensors="pt")

    output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50,
                            top_p=0.95, temperature=0.1)

    generated_comments = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_comments


# Example usage
code_snippet = """

PLEASE ADD COMMENTS TO THIS PYTHON CODE.

def add(a, b):
    return a + b
"""

comment = generate_comment(code_snippet)
print(comment)
