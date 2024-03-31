def problema4(text: str) -> list[str]:
    dictionar = dict()
    for word in text.split(" "):
        dictionar[word] = dictionar.get(word, 0) + 1

    return [word for word, count in dictionar.items() if count == 1]


assert (problema4("ana are ana are mere rosii ana") == ["mere", "rosii"])
assert (problema4("ana are ana are") == [])