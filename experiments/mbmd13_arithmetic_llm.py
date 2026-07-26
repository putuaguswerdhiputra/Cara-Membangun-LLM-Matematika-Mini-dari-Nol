# ==============================================
# MBMD-13: Arithmetic LLM Mini (Transformer Decoder)
# ==============================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, math, time, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class ArithmeticDataset(Dataset):
    def __init__(self, n_samples=2000000, max_digits=4, ops=['+','-','*'], seed=42):
        np.random.seed(seed); self.data=[]
        for _ in range(n_samples):
            op=np.random.choice(ops)
            if op=='+': a=np.random.randint(0,10**max_digits); b=np.random.randint(0,10**max_digits); result=a+b
            elif op=='-': a=np.random.randint(0,10**max_digits); b=np.random.randint(0,a+1); result=a-b
            elif op=='*': a=np.random.randint(0,10**max_digits); b=np.random.randint(0,min(100,10**max_digits)); result=a*b
            else: continue
            self.data.append((f"{a}{op}{b}=", f"{result}"))
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

chars=[str(i) for i in range(10)]+['+','-','*','=','<EOS>','<PAD>']
stoi={ch:i for i,ch in enumerate(chars)}; itos={i:ch for i,ch in enumerate(chars)}
vocab_size=len(chars); pad_token=stoi['<PAD>']; eos_token=stoi['<EOS>']

def encode_sequence(text,max_len):
    tokens=[stoi.get(c,pad_token) for c in text]
    if len(tokens)>max_len: tokens=tokens[:max_len]
    else: tokens+=[pad_token]*(max_len-len(tokens))
    return tokens

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=20):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(position*div_term); pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.size(1)]

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=20):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size,d_model)
        self.pos_encoder=PositionalEncoding(d_model,max_len)
        decoder_layer=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,batch_first=True)
        self.transformer=nn.TransformerDecoder(decoder_layer,num_layers)
        self.fc_out=nn.Linear(d_model,vocab_size); self.max_len=max_len; self.d_model=d_model
    def forward(self,x):
        seq_len=x.size(1)
        x=self.token_embed(x)*math.sqrt(self.d_model); x=self.pos_encoder(x)
        mask=torch.triu(torch.ones(seq_len,seq_len,device=x.device)*float('-inf'),diagonal=1)
        out=self.transformer(x,x,tgt_mask=mask); out=self.fc_out(out); return out

def collate_fn(batch):
    max_input_len=20; max_target_len=20; inputs_list=[]; targets_list=[]
    for inp_str,out_str in batch:
        full_seq=inp_str+out_str+'<EOS>'
        seq_ids=encode_sequence(full_seq,max_input_len)
        input_ids=seq_ids[:-1]; target_ids=seq_ids[1:]
        if len(input_ids)<max_input_len-1:
            input_ids+=[pad_token]*(max_input_len-1-len(input_ids))
            target_ids+=[pad_token]*(max_target_len-1-len(target_ids))
        inputs_list.append(input_ids); targets_list.append(target_ids)
    return torch.tensor(inputs_list),torch.tensor(targets_list)

def train(model,epochs=15,lr=1e-3,warmup=500):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95))
    criterion=nn.CrossEntropyLoss(ignore_index=pad_token)
    scheduler=optim.lr_scheduler.LambdaLR(optimizer,lambda step: min((step+1)/warmup,1.) if step<warmup else 0.5*(1+math.cos(math.pi*(step-warmup)/(epochs*len(train_loader)-warmup))))
    history={'loss':[],'acc':[]}
    for ep in range(epochs):
        model.train(); total_loss=0.; total_correct=0; total_tokens=0; t0=time.time()
        for step,(inputs,targets) in enumerate(train_loader):
            inputs,targets=inputs.to(device),targets.to(device)
            logits=model(inputs); loss=criterion(logits.view(-1,vocab_size),targets.view(-1))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            scheduler.step()
            total_loss+=loss.item()*inputs.size(0)
            preds=logits.argmax(dim=-1); mask=(targets!=pad_token)
            correct=((preds==targets)&mask).sum().item(); total_tokens+=mask.sum().item(); total_correct+=correct
            if step%500==0: print(f"Ep {ep+1} Step {step:5d} | Loss: {loss.item():.4f} | Acc: {correct/mask.sum().item() if mask.sum()>0 else 0:.4f}")
        avg_loss=total_loss/len(train_loader.dataset); avg_acc=total_correct/total_tokens
        history['loss'].append(avg_loss); history['acc'].append(avg_acc)
        print(f"=== Epoch {ep+1:2d} | Loss: {avg_loss:.4f} | Token Acc: {avg_acc:.4f} | Time: {time.time()-t0:.1f}s")
        if (ep+1)%5==0: torch.save(model.state_dict(),f'arithmetic_mini_ep{ep+1}.pt')
    return history

dataset=ArithmeticDataset(n_samples=2000000,max_digits=4)
train_loader=DataLoader(dataset,batch_size=256,shuffle=True,collate_fn=collate_fn,num_workers=2)
model=MiniGPT(vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=20)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
history=train(model,epochs=15,lr=1e-3,warmup=500)

def evaluate(num_samples=1000,max_digits=4):
    model.eval()
    test_dataset=ArithmeticDataset(n_samples=num_samples,max_digits=max_digits,seed=999); correct=0
    with torch.no_grad():
        for inp_str,out_str in test_dataset.data:
            full_prompt=inp_str; generated=""
            for _ in range(10):
                seq=full_prompt+generated; seq_ids=encode_sequence(seq,20)
                input_tensor=torch.tensor([seq_ids[:-1]],device=device)
                logits=model(input_tensor); next_logit=logits[0,-1,:]
                next_token=torch.argmax(torch.softmax(next_logit,dim=-1)).item()
                if itos[next_token]=='<EOS>': break
                generated+=itos[next_token]
            if generated==out_str: correct+=1
    acc=correct/num_samples; print(f"Test Exact Match Accuracy: {acc:.4f}"); return acc

test_acc=evaluate(num_samples=1000)
