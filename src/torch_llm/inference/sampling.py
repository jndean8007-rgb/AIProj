import torch as t

def sample(
        logits,
        tokenizer
): #can be more complicated later

    indices = t.argmax(logits, dim=-1)

    return tokenizer.decode(list(indices))