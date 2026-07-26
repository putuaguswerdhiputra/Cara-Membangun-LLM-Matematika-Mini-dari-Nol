# ==============================================
# MBMD-14: Reasoning-Enhanced Arithmetic LLM Mini
# ==============================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np, math, time, matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

special_tokens=['<s>','</s>','<step>','</step>','<result>','</result>','<EOS>','<PAD>']
digit_tokens=[str(i) for i in range(10)]; operator_tokens=['+','-','*','=']
other_tokens=[' ','c','a','r','y']
all_chars=list(dict.fromkeys(digit_tokens+operator_tokens+other_tokens+special_tokens))
stoi={ch:i for i,ch in enumerate(all_chars)}; itos={i:ch for ch,i in stoi.items()}
vocab_size=len(stoi); pad_token=stoi['<PAD>']; eos_token=stoi['<EOS>']
print(f"Vocab size: {vocab_size}")

def encode_sequence(text,max_len):
    tokens=[stoi.get(c,pad_token) for c in text]
    if len(tokens)>max_len: tokens=tokens[:max_len]
    else: tokens+=[pad_token]*(max_len-len(tokens))
    return tokens

class ArithmeticReasoningDataset(Dataset):
    def __init__(self,n_samples=100000,max_digits=3,seed=42):
        np.random.seed(seed); self.data=[]
        for _ in range(n_samples):
            d=np.random.randint(1,max_digits+1); a=np.random.randint(0,10**d); b=np.random.randint(0,10**d); result=a+b
            reasoning=self._generate_reasoning(a,b,d)
            input_str=f"<s>{a}+{b}=</s>"
            reasoning_str=''.join([f"<step>{step}</step>" for step in reasoning])
            result_str=f"<result>{result}</result><EOS>"; full_output=reasoning_str+result_str
            self.data.append((input_str,full_output))

    def _generate_reasoning(self,a,b,digits):
        str_a=str(a).zfill(digits); str_b=str(b).zfill(digits); carry=0; steps=[]
        for i in range(digits-1,-1,-1):
            da=int(str_a[i]); db=int(str_b[i]); s=da+db+carry; digit_result=s%10; carry=s//10
            step=f"{da}+{db}+{carry if i<digits-1 else 0}={s} carry{carry}"; steps.append(step)
        if carry>0: steps.append(f"carry={carry}")
        return steps
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=128):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1).float()
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(position*div_term); pe[:,1::2]=torch.cos(position*div_term)
        self.register_buffer('pe',pe)
    def forward(self,x): return x+self.pe[:x.size(1)]

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=128):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size,d_model); self.pos_encoder=PositionalEncoding(d_model,max_len)
        decoder_layer=nn.TransformerDecoderLayer(d_model,nhead,dim_feedforward,batch_first=True)
        self.transformer=nn.TransformerDecoder(decoder_layer,num_layers)
        self.fc_out=nn.Linear(d_model,vocab_size); self.max_len=max_len; self.d_model=d_model
    def forward(self,x):
        seq_len=x.size(1); x=self.token_embed(x)*math.sqrt(self.d_model); x=self.pos_encoder(x)
        mask=torch.triu(torch.ones(seq_len,seq_len,device=x.device)*float('-inf'),diagonal=1)
        out=self.transformer(x,x,tgt_mask=mask); out=self.fc_out(out); return out

def collate_fn(batch):
    max_total_len=128; inputs=[]; targets=[]
    for inp_str,out_str in batch:
        full=inp_str+out_str; seq_ids=encode_sequence(full,max_total_len)
        inputs.append(seq_ids[:-1]); targets.append(seq_ids[1:])
    return torch.tensor(inputs),torch.tensor(targets)

def train(model,train_loader,epochs=15,lr=1e-3,warmup=500):
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    criterion=nn.CrossEntropyLoss(ignore_index=pad_token)
    total_steps=epochs*len(train_loader)
    scheduler=optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=total_steps,pct_start=warmup/total_steps,anneal_strategy='cos')
    history={'loss':[],'acc':[],'time':[]}
    for ep in range(epochs):
        model.train(); total_loss=0.; total_correct=0; total_tokens=0; t0=time.time()
        for step,(inputs,targets) in enumerate(train_loader):
            inputs,targets=inputs.to(device),targets.to(device)
            logits=model(inputs); loss=criterion(logits.view(-1,vocab_size),targets.view(-1))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
            total_loss+=loss.item()*inputs.size(0)
            preds=logits.argmax(dim=-1); mask=(targets!=pad_token)
            correct=((preds==targets)&mask).sum().item(); total_tokens+=mask.sum().item(); total_correct+=correct
            if step%200==0: print(f"Ep {ep+1} Step {step:5d} | Loss: {loss.item():.4f} | Token Acc: {correct/mask.sum().item() if mask.sum()>0 else 0:.4f}")
        avg_loss=total_loss/len(train_loader.dataset); avg_acc=total_correct/total_tokens
        history['loss'].append(avg_loss); history['acc'].append(avg_acc); history['time'].append(time.time()-t0)
        print(f"=== Epoch {ep+1:2d} | Loss: {avg_loss:.4f} | Token Acc: {avg_acc:.4f} | Time: {time.time()-t0:.1f}s")
        if (ep+1)%5==0: torch.save(model.state_dict(),f'mbmd14_ep{ep+1}.pt')
    return history

def evaluate_and_analyze(num_samples=100):
    model.eval()
    test_set=ArithmeticReasoningDataset(n_samples=num_samples,max_digits=3,seed=999)
    correct_result=0; correct_full=0; total=0
    with torch.no_grad():
        for inp_str,out_str in test_set.data:
            full_prompt=inp_str; generated=""
            for _ in range(100):
                current=full_prompt+generated
                seq_ids=torch.tensor([encode_sequence(current,128)[:-1]],device=device)
                logits=model(seq_ids); next_token=torch.argmax(torch.softmax(logits[0,-1,:],dim=-1)).item()
                if next_token==eos_token: break
                generated+=itos[next_token]
            if '<result>' in generated and '</result>' in generated:
                start=generated.find('<result>')+len('<result>'); end=generated.find('</result>')
                pred_result=generated[start:end]; true_result=out_str.split('<result>')[1].split('</result>')[0]
                if pred_result==true_result: correct_result+=1
                if generated+'<EOS>'==out_str: correct_full+=1
            total+=1
    print(f"Exact result match: {correct_result}/{total} = {correct_result/total:.4f}")
    print(f"Full sequence match: {correct_full}/{total} = {correct_full/total:.4f}")

train_dataset=ArithmeticReasoningDataset(n_samples=200000,max_digits=3)
train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True,collate_fn=collate_fn,num_workers=2)
model=MiniGPT(vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=128)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
history=train(model,train_loader,epochs=20,lr=1e-3,warmup=500)
evaluate_and_analyze(num_samples=200)
