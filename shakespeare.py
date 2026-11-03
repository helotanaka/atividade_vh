import os
import urllib.request
import numpy as np
import tiktoken

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  #texto do Tiny Shakespeare (cerca de 1 MB)
data_root = "edu_fineweb10B"  #mesma pasta que o DataLoaderLite lê
os.makedirs(data_root, exist_ok=True)

#baixa o texto
txt_path = "input.txt"
if not os.path.exists(txt_path):
    print("baixando o Tiny Shakespeare...")
    urllib.request.urlretrieve(url, txt_path)
with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

#tokeniza com o mesmo tokenizador do GPT-2
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  #token especial que marca o início do documento
tokens = [eot] + enc.encode_ordinary(text)
tokens_np = np.array(tokens)
assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()  #garante que cabe em uint16
tokens_np = tokens_np.astype(np.uint16)  #uint16 economiza espaço, o vocabulário tem menos de 65536 tokens

#separa 90% para treino e 10% para validação
n = int(0.9 * len(tokens_np))
train_tokens = tokens_np[:n]
val_tokens = tokens_np[n:]

#os nomes precisam conter "train" ou "val", pois o DataLoaderLite filtra por isso
np.save(os.path.join(data_root, "edufineweb_val_000000"), val_tokens)
np.save(os.path.join(data_root, "edufineweb_train_000001"), train_tokens)

print(f"total de tokens: {len(tokens_np):,}")
print(f"treino: {len(train_tokens):,} tokens | validação: {len(val_tokens):,} tokens")
print(f"arquivos salvos em {data_root}/")