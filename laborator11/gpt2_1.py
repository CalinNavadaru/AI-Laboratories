from transformers import GPT2Tokenizer, GPT2LMHeadModel, EarlyStoppingCallback

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

with open('code.txt', 'r', encoding='utf-8') as file:
    text = file.read()

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler


class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.texts[idx]
        inputs = self.tokenizer(item, max_length=self.max_length, truncation=True, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in inputs.items()}


dataset = CustomDataset([text], tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = 1000
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.train()

for epoch in range(12):
    for batch in dataloader:
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f"Loss: {loss.item()}")

model.save_pretrained('fine-tuned-gpt2')
tokenizer.save_pretrained('fine-tuned-gpt2')

model = GPT2LMHeadModel.from_pretrained('fine-tuned-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('fine-tuned-gpt2')

d = """
def problema3(vector1: list[float], vector2: list[float]) -> float:
    rez = 0
    for a, b in zip(vector1, vector2):
"""
inputs = tokenizer(d, return_tensors="pt", truncation=True,
                   max_length=256)
outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, pad_token_id=tokenizer.eos_token_id,
                         max_length=250, num_return_sequences=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
