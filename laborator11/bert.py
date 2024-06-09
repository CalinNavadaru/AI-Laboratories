import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

#
#
clase = {0: "functie",
         1: "for",
         2: "if",
         3: "set instructiuni secventiale"}
#
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=len(clase))
#
#
# def preprocess_function(example):
#     tokenized_code = tokenizer(example["code"], truncation=True, padding="max_length", max_length=512)
#
#     label = torch.tensor(example["label"])
#
#     return {**tokenized_code, "labels": label}
#
#
# train_dataset = [
#     {"code": "def add(a, b):\n    return a + b", "label": 0},
#     {"code": "for i in range(10):\n    print(i)", "label": 1},
#     {"code": """def problema1(text: str) -> str:
#     last_word = ""
#     for word in text.split(" "):
#         if last_word < word:
#             last_word = word
#
#     return last_word""", "label": 0},
#     {"code": """if a > 5:
#         print('da')
#     else
#         print('nu')""", "label": 2},
#     {"code": "print(5 + 2)", "label": 3},
#     {"code": """s = 0
#     for i in range(5):
#         s += i
#     print(s)
#     """, "label": 1},
#     {"code": "print('a' if 'a' > 'b' else 'b')", "label": 3},
#     {"code": "def subtract(a, b):\n\teturn a - b", "label": 0},
#     {"code": "def cube(a):\n    return a * a * a", "label": 0},
#     {"code": "for i in range(5):\n    print(i ** 2)", "label": 1},
#     {"code": """def problem5(text: str) -> str:
#     words = text.split(" ")
#     shortest_word = min(words, key=len)
#
#     return shortest_word""", "label": 0},
#     {"code": """if a != 5:
#         print('not equal')
#     else:
#         print('equal')""", "label": 2},
#     {"code": "print(5 / 2)", "label": 3},
#     {"code": "print(2)", "label": 3},
#     {"code": "print(5 + 2 + 2 - 0)", "label": 3},
#     {"code": """if 5 + 5 > 9:
#         return 10
#     else:
#         return 5""", "label": 2},
#     {"code": """print('AI-ul este cool domne')""", "label": 3},
#     {"code": """a = 5
#     b = 2
#     print(a + b)
#     print(a - b)
#     print(a * b)
#     print(a / b)""", "label": 3},
#     {"code": """def f(x: list[int]):
#         p = 1
#         for y in x:
#             p *= y
#         return p""", "label": 0}
# ]
# valid_dataset = [
#     {"code": "def multiply(a, b):\n    return a * b", "label": 0},
#     {"code": "for i in range(5):\n    print(i * 0)", "label": 1},
#     {"code": """def problema3(text: str) -> str:
#     first_word = ""
#     for word in text.split(" "):
#         if first_word < word:
#             first_word = word
#
#     return first_word""", "label": 0},
#     {"code": """if a < 5:
#         print('yes')
#     else:
#         print('no')""", "label": 2},
#     {"code": "print(5 - 2)", "label": 3},
#     {"code": """s = 0
#     for i in range(2):
#         s += i
#     print(s)
#     """, "label": 1},
#     {"code": "print('b' if 'a' < 'b' else 'a')", "label": 2},
#     {"code": "def divide(a, b):\n\teturn a / b", "label": 0},
#     {"code": "def cube(a):\n    return a * a * a", "label": 0},
#     {"code": "for i in range(5):\n    print(i ** 2)", "label": 1},
#     {"code": """def problem5(text: str) -> str:
#     words = text.split(" ")
#     shortest_word = min(words, key=len)
#
#     return shortest_word""", "label": 0},
#     {"code": """if a != 5:
#         print('not equal')
#     else:
#         print('equal')""", "label": 2},
#     {"code": "print(5 / 2)", "label": 3},
# ]
#
# train_dataset = [preprocess_function(example) for example in train_dataset]
# valid_dataset = [preprocess_function(example) for example in valid_dataset]
#
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=12,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset
# )
#
# trainer.train()
#
# tokenizer.save_pretrained("./model")
# model.save_pretrained("./model")

tokenizer = RobertaTokenizer.from_pretrained("./model")
model = RobertaForSequenceClassification.from_pretrained("./model", num_labels=len(clase))

prompt = """print(3)"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model(**inputs)

logits = outputs.logits


probs = torch.nn.functional.softmax(logits, dim=-1)

predicted_class = torch.argmax(probs, dim=-1).item()

print(f"Predicted class: {predicted_class}")
print(clase[predicted_class])
