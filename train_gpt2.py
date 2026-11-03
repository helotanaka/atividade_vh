import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  #gera query, key e value de todas as cabeças de uma só vez
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  #junta as informações das cabeças
        self.c_proj.NANOGPT_SCALE_INIT = 1  #identificação da camada durante a inicialização dos pesos
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  #tamanho do batch, comprimento(tokens) e dimensão do embedding(channels)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  #separação em múltiplas cabeças (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  #flash attention com máscara causal
        y = y.transpose(1, 2).contiguous().view(B, T, C)  #junta a saída de todas as cabeças lado a lado
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)  #aumenta para facilitar os cálculos
        self.gelu    = nn.GELU(approximate='tanh')  #introduz não linearidade para ajudar o modelo a aprender padrões complexos
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)  #reduz a dimensão ao tamanho original
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  #conexão residual: soma a entrada com o resultado da atenção, ajuda o gradiente a fluir
        x = x + self.mlp(self.ln_2(x))  #conexão residual: soma a entrada com o resultado da MLP
        return x

@dataclass
#tamanho e a estrutura do modelo
class GPTConfig:
    block_size: int = 1024  #tamanho máximo da sequência (contexto)
    vocab_size: int = 50257  #número de tokens: 50.000 merges do BPE + 256 tokens de bytes + 1 token <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  #transforma cada token em um vetor de embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  #representa a posição de cada token na sequência
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  #contém os blocos Transformer que processam as representações
            ln_f = nn.LayerNorm(config.n_embd),  #normaliza a representação final
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  #transforma a representação final em pontuações para os tokens do vocabulário

        self.transformer.wte.weight = self.lm_head.weight  #compartilha os pesos entre embedding de entrada e camada de saída

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  #reduz o desvio nas camadas residuais para o valor não crescer com a profundidade
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()  #idx tem formato (B, T): batch e comprimento da sequência
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb  #soma token e posição para o modelo saber o que é e onde está cada token
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  #calcula as pontuações de cada token do vocabulário, formato (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  #calcula o erro comparando as previsões com os tokens corretos
        return logits, loss
        #Logits são os valores numéricos que o modelo produz para representar o quanto cada token do vocabulário é provável de ser o próximo token

    @classmethod
    def from_pretrained(cls, model_type):
    #Carrega os pesos pré-treinados do GPT-2 do huggingface
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        #camadas, cabeças e embedding dependem do tamanho escolhido
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M parâmetros
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M parâmetros
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M parâmetros
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parâmetros
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  #descarta a máscara causal, que é um buffer e não um parâmetro

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)  #carrega o modelo original do huggingface
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']  #pesos que precisam ser transpostos
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  #copia o peso transposto, sem calcular gradiente
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # separa em dois grupos: parâmetros com 2 ou mais dimensões sofrem weight decay, os outros não
        # ou seja, pesos de matrizes e embeddings decaem, biases e layernorms não
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"  #só usa a versão fused (mais rápida) se estiver na GPU
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        #busca os arquivos (shards) do dataset
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]  #filtra os arquivos do split escolhido
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank  #cada processo começa em uma posição diferente para não repetir dados

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]  #pega B*T tokens + 1 extra para formar os alvos
        x = (buf[:-1]).view(B, T)  #entradas: todos os tokens menos o último
        y = (buf[1:]).view(B, T)  #alvos: os mesmos tokens deslocados uma posição (o próximo token)
        self.current_position += B * T * self.num_processes
        # se o próximo batch passar do fim do arquivo, vai para o próximo shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# função auxiliar para a avaliação do HellaSwag
# recebe tokens, máscara e logits e retorna o índice da continuação com menor loss

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()  #remove o último logit, pois não tem token seguinte para comparar
    shift_tokens = (tokens[..., 1:]).contiguous()  #remove o primeiro token, que não é previsto por ninguém
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')  #loss de cada posição, sem fazer a média
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()  #desloca a máscara também, para começar no último token do contexto
    masked_shift_losses = shift_losses * shift_mask  #zera o loss da parte do contexto, fica só a continuação
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)  #divide pelo número de tokens da continuação para obter a média
    # agora temos o loss médio de cada uma das 4 continuações
    # a de menor loss é a mais provável
    pred_norm = avg_loss.argmin().item()  #índice da continuação com menor loss
    return pred_norm

# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
# execução simples

# a partir daqui começa o loop de treinamento
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# configuração do DDP (distributed data parallel), que treina em várias GPUs ao mesmo tempo
# o comando torchrun define as variáveis de ambiente RANK, LOCAL_RANK e WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  #se RANK existe, é uma execução com DDP
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'  #cada processo usa a sua própria GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  #só o processo 0 faz prints, logs e salva checkpoints
else:
    # execução normal, sem DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #detecta automaticamente o melhor dispositivo disponível
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)  #semente fixa para os resultados serem reproduzíveis
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 32768  #2**15, reduzido para teste (o original era 2**19, cerca de 0,5M tokens)
B = 4  #micro batch size, quantas sequências passam de uma vez pela GPU
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)  #quantos micro batches acumular para chegar ao batch total
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')  #permite multiplicações de matrizes mais rápidas com precisão um pouco menor (TF32)

model = GPT(GPTConfig(vocab_size=50304))  #cria o modelo do zero, vocabulário de 50304 (múltiplo de 128) para ficar mais eficiente na GPU
# model = GPT.from_pretrained("gpt2")  #alternativa: iniciar com os pesos da OpenAI
model.to(device)
use_compile = False  #torch.compile atrapalha a avaliação do HellaSwag e a geração de texto
use_hellaswag = False  #desligado no teste, pois avalia ~10 mil exemplos e demora muito
eval_interval = 50  #a cada quantos passos avalia e gera texto (o original era 250)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])  #envolve o modelo para sincronizar os gradientes entre as GPUs
raw_model = model.module if ddp else model  #sempre guarda o modelo original, sem o envoltório do DDP

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10  #passos iniciais em que a taxa de aprendizado sobe aos poucos (o original era 715)
max_steps = 200  #poucos passos para teste (o original era 19073, cerca de 1 época com 10B de tokens)
def get_lr(it):
    # 1) warmup linear durante os warmup_steps primeiros passos
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) depois do último passo, mantém a taxa mínima
    if it > max_steps:
        return min_lr
    # 3) entre os dois, usa decaimento cosseno até a taxa mínima
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  #o coeficiente começa em 1 e vai até 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # de tempos em tempos avalia o loss de validação
    if step % eval_interval == 0 or last_step:
        model.eval()  #modo de avaliação, desliga comportamentos de treino
        val_loader.reset()
        with torch.no_grad():  #sem calcular gradientes, economiza memória
            val_loss_accum = 0.0
            val_loss_steps = 4  #número de batches usados para estimar o loss (poucos, pois a validação é pequena)
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  #usa bfloat16 nas contas para ficar mais rápido
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                #salva um checkpoint do modelo
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # de tempos em tempos avalia no HellaSwag
    if (step % eval_interval == 0 or last_step) and (not use_compile) and use_hellaswag:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # cada processo avalia só os exemplos em que i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)  #transforma o exemplo em tokens, máscara e resposta correta
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)  #escolhe a continuação com menor loss
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # soma as estatísticas de todos os processos
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total  #acurácia = acertos divididos pelo total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # de tempos em tempos gera texto com o modelo (exceto no passo 0, que só gera ruído)
    if ((step > 0 and step % eval_interval == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4  #quantos textos gerar
        max_length = 32  #tamanho máximo de cada texto em tokens
        tokens = enc.encode("Hello, I'm a language model,")  #texto inicial (prompt) convertido em tokens
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]  #fica só com os logits da última posição, que preveem o próximo token (B, vocab_size)
                probs = F.softmax(logits, dim=-1)  #transforma os logits em probabilidades
                # amostragem top-k com k=50 (padrão do pipeline do huggingface)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  #mantém só os 50 tokens mais prováveis
                # sorteia um token entre os top-k de acordo com as probabilidades
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)  #pega o índice real do token no vocabulário (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)  #adiciona o novo token ao final da sequência
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # executa um passo de otimização
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  #só sincroniza os gradientes entre as GPUs no último micro passo
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # é preciso dividir o loss por causa do acúmulo de gradientes,
        # pois os gradientes são somados a cada backward()
        # somar gradientes equivale a uma SOMA no objetivo, mas queremos a MÉDIA
        # então o loss é dividido aqui para o resultado sair certo
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()  #calcula os gradientes (backpropagation)
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #limita a norma dos gradientes a 1.0 para evitar atualizações muito grandes
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()  #atualiza os pesos do modelo
    if device_type == "cuda":
        torch.cuda.synchronize()  #espera a GPU terminar o trabalho para medir o tempo corretamente
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()