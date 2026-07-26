# ==============================================
# MBMD-16: Clean Arithmetic Solver (tanpa bahasa alami)
# ==============================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, math, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

special_tokens=['<s>','</s>','<step>','</step>','<result>','</result>','<EOS>','<PAD>','<tool>','</tool>','<python>','</python>']
digit_tokens=[str(i) for i in range(10)]; operator_tokens=['+','-','*','=']
all_chars=digit_tokens+operator_tokens+special_tokens
stoi={ch:i for i,ch in enumerate(all_chars)}; itos={i:ch for ch,i in stoi.items()}
vocab_size=len(stoi); pad_token=stoi['<PAD>']; eos_token=stoi['<EOS>']
print(f"Vocab size: {vocab_size}")

def encode_sequence(text,max_len):
    tokens=[stoi.get(c,pad_token) for c in text]
    if len(tokens)>max_len: tokens=tokens[:max_len]
    else: tokens+=[pad_token]*(max_len-len(tokens))
    return tokens

class CleanArithmeticDataset(Dataset):
    def __init__(self,n_samples=200000,max_digits=4,tool_threshold=3):
        np.random.seed(42); self.data=[]
        for _ in range(n_samples):
            op=np.random.choice(['+','-','*'])
            max_d=max_digits if op!='*' else min(max_digits,3)
            d1=np.random.randint(1,max_d+1); d2=np.random.randint(1,max_d+1)
            a=np.random.randint(0,10**d1); b=np.random.randint(0,10**d2)
            if op=='-' and a<b: a,b=b,a
            result=eval(f"{a}{op}{b}")
            use_tool=(max(d1,d2)>=tool_threshold) or (op=='*' and a*b>10**(tool_threshold*2))
            input_str=f"<s>{a}{op}{b}=</s>"
            if use_tool:
                output_str=f"<tool><python>print({a}{op}{b})</python></tool><result>{result}</result><EOS>"
            else:
                output_str=self._generate_manual_steps(a,b,op)+f"<result>{result}</result><EOS>"
            self.data.append((input_str,output_str))

    def _generate_manual_steps(self,a,b,op):
        if op=='+':
            str_a=str(a).zfill(max(len(str(a)),len(str(b)))); str_b=str(b).zfill(max(len(str(a)),len(str(b))))
            carry=0; steps=[]
            for i in range(len(str_a)-1,-1,-1):
                da=int(str_a[i]); db=int(str_b[i]); s=da+db+carry; carry=s//10
                steps.append(f"<step>{da}+{db}+{carry if i<len(str_a)-1 else 0}={s} carry{carry}</step>")
            if carry>0: steps.append(f"<step>carry={carry}</step>")
            return ''.join(steps)
        elif op=='-':
            str_a=str(a).zfill(len(str(b))); str_b=str(b).zfill(len(str(a)))
            borrow=0; steps=[]
            for i in range(len(str_a)-1,-1,-1):
                da=int(str_a[i])-borrow; db=int(str_b[i])
                if da<db: da+=10; borrow=1
                else: borrow=0
                steps.append(f"<step>{da}-{db}={da-db} borrow{borrow}</step>")
            return ''.join(steps)
        else: return f"<step>{a}*{b}=?</step>"
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=256):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(position*div_term); pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.size(1)]

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=256):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size,d_model); self.pos_encoder=PositionalEncoding(d_model,max_len)
        decoder_layer=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,batch_first=True)
        self.transformer=nn.TransformerDecoder(decoder_layer,num_layers)
        self.fc_out=nn.Linear(d_model,vocab_size); self.max_len=max_len; self.d_model=d_model
    def forward(self,x):
        x=self.token_embed(x)*math.sqrt(self.d_model); x=self.pos_encoder(x)
        mask=torch.triu(torch.ones(x.size(1),x.size(1),device=x.device)*float('-inf'),diagonal=1)
        out=self.transformer(x,x,tgt_mask=mask); return self.fc_out(out)

def collate_fn(batch,max_len=256):
    inputs,targets=[],[]
    for inp,out in batch:
        seq=encode_sequence(inp+out,max_len)
        inputs.append(seq[:-1]); targets.append(seq[1:])
    return torch.tensor(inputs),torch.tensor(targets)

def train_model(model,train_loader,epochs=15,lr=1e-3):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr)
    criterion=nn.CrossEntropyLoss(ignore_index=pad_token)
    for ep in range(epochs):
        model.train(); total_loss=0.; t0=time.time()
        for inputs,targets in train_loader:
            inputs,targets=inputs.to(device),targets.to(device)
            logits=model(inputs); loss=criterion(logits.view(-1,vocab_size),targets.view(-1))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            total_loss+=loss.item()*inputs.size(0)
        avg_loss=total_loss/len(train_loader.dataset)
        print(f"Epoch {ep+1:2d} | Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")
    torch.save(model.state_dict(),'mbmd16_clean.pt'); print("Model saved.")

def generate_simple(model,prompt,max_len=256):
    model.eval()
    with torch.no_grad():
        current=prompt
        for _ in range(max_len):
            seq_ids=torch.tensor([encode_sequence(current,max_len)[:-1]],device=device)
            logits=model(seq_ids); next_token=torch.argmax(logits[0,-1,:]).item()
            if next_token==eos_token: break
            current+=itos[next_token]
    return current

train_dataset=CleanArithmeticDataset(n_samples=150000,max_digits=4)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,collate_fn=lambda b: collate_fn(b))
model=MiniGPT(vocab_size)
train_model(model,train_loader,epochs=15)

np.random.seed(999)
test_data=[(f"<s>{a}{op}{b}=</s>",f"{a}{op}{b}") for a,b,op in [(24,9,'*'),(683,158,'-'),(442,120,'+')]]
for prompt,true_expr in test_data:
    true_result=eval(true_expr); pred=generate_simple(model,prompt)
    print(f"\nPrompt: {prompt}\nTrue : {true_expr} = {true_result}\nPred : {pred}")
