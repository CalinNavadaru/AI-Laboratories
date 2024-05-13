import random


def build_chain(text, n):
    words = text.split()
    chain = {}
    for i in range(len(words) - n):
        key = tuple(words[i:i + n])
        value = words[i + n]
        if key in chain:
            chain[key].append(value)
        else:
            chain[key] = [value]
    return chain


def generate_text(chain, n, length):
    words = random.choice(list(chain.keys()))
    result = list(words)
    for i in range(length):
        key = tuple(words)
        next_word = random.choice(chain[key])
        result.append(next_word)
        words = result[-n:]
    return ' '.join(result)


with open("data/corpus_complet.txt", encoding='utf-8') as f:
    text = f.read()

chain = build_chain(text, 1)
result_text = generate_text(chain, 1, 200).split(".")
for prop in result_text:
    print(prop)

chain = build_chain(text, 4)
with open("data/corpus_complet.txt", encoding='utf-8') as f:
    text = f.read()

result_text = generate_text(chain, 4, 200).split(".")
for prop in result_text:
    print(prop)
