# ==============================================
# MBMD-15: Neuro-Symbolic Arithmetic LLM Mini
# ==============================================

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, math, time, re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

special_tokens=['<s>','</s>','<step>','</step>','<result>','</result>','<EOS>','<PAD>','<tool>','</tool>','<python>','</python>']
digit_tokens=[str(i) for i in range(10)]; operator_tokens=['+','-','*','=']
letter_tokens=list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
all_chars=list(dict.fromkeys(digit_tokens+operator_tokens+letter_tokens+special_tokens))
stoi={ch:i for i,ch in enumerate(all_chars)}; itos={i:ch for ch,i in stoi.items()}
vocab_size=len(stoi); pad_token=stoi['<PAD>']; eos_token=stoi['<EOS>']
print(f"Vocab size: {vocab_size}")

def encode_sequence(text,max_len):
    tokens=[stoi.get(c,pad_token) for c in text]
    if len(tokens)>max_len: tokens=tokens[:max_len]
    else: tokens+=[pad_token]*(max_len-len(tokens))
    return tokens

class NeuroSymbolicArithmeticDataset(Dataset):
    def __init__(self,n_samples=200000,max_digits=4,tool_threshold=3,seed=42):
        np.random.seed(seed); self.data=[]
        templates=["{a}{op}{b}=","Berapakah {a} {op} {b}?","Hitung {a} {op} {b}.","Jumlahkan {a} dan {b}","{a} {op} {b} sama dengan","Tentukan hasil dari {a} {op} {b}"]
        for _ in range(n_samples):
            op=np.random.choice(['+','-','*'])
            max_d=max_digits if op!='*' else min(max_digits,3)
            d1=np.random.randint(1,max_d+1); d2=np.random.randint(1,max_d+1)
            a=np.random.randint(0,10**d1); b=np.random.randint(0,10**d2)
            if op=='-' and a<b: a,b=b,a
            result=eval(f"{a}{op}{b}")
            use_tool=(max(d1,d2)>=tool_threshold) or (op=='*' and a*b>10**(tool_threshold*2))
            template=np.random.choice(templates)
            if '{op}' in template:
                op_display={'+':'+','-':'-','*':'×'}.get(op,op)
                question=template.format(a=a,op=op_display,b=b)
            else: question=template.format(a=a,b=b)
            if len(question)>60: question=f"{a}{op}{b}="
            if use_tool: reasoning_str=f"<tool><python>print({a}{op}{b})</python></tool>"
            else: reasoning_str=self._generate_manual_steps(a,b,op)
            full_output=f"{reasoning_str}<result>{result}</result><EOS>"
            input_str=f"<s>{question}</s>"
            self.data.append((input_str,full_output))

    def _generate_manual_steps(self,a,b,op):
        if op=='+': return self._addition_steps(a,b)
        elif op=='-': return self._subtraction_steps(a,b)
        else: return f"<step>{a}*{b}=?</step>"
    def _addition_steps(self,a,b):
        str_a=str(a).zfill(max(len(str(a)),len(str(b)))); str_b=str(b).zfill(max(len(str(a)),len(str(b))))
        carry=0; steps=[]
        for i in range(len(str_a)-1,-1,-1):
            da=int(str_a[i]); db=int(str_b[i]); s=da+db+carry; carry=s//10
            steps.append(f"<step>{da}+{db}+{carry if i<len(str_a)-1 else 0}={s} carry{carry}</step>")
        if carry>0: steps.append(f"<step>carry={carry}</step>")
        return ''.join(steps)
    def _subtraction_steps(self,a,b):
        str_a=str(a).zfill(len(str(b))); str_b=str(b).zfill(len(str(a)))
        borrow=0; steps=[]
        for i in range(len(str_a)-1,-1,-1):
            da=int(str_a[i])-borrow; db=int(str_b[i])
            if da<db: da+=10; borrow=1
            else: borrow=0
            steps.append(f"<step>{da}-{db}={da-db} borrow{borrow}</step>")
        return ''.join(steps)
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
        seq_len=x.size(1); x=self.token_embed(x)*math.sqrt(self.d_model); x=self.pos_encoder(x)
        mask=torch.triu(torch.ones(seq_len,seq_len,device=x.device)*float('-inf'),diagonal=1)
        out=self.transformer(x,x,tgt_mask=mask); out=self.fc_out(out); return out

def collate_fn(batch):
    max_total_len=256; inputs=[]; targets=[]
    for inp_str,out_str in batch:
        full=inp_str+out_str; seq_ids=encode_sequence(full,max_total_len)
        inputs.append(seq_ids[:-1]); targets.append(seq_ids[1:])
    return torch.tensor(inputs),torch.tensor(targets)

def train(model,train_loader,epochs=20,lr=1e-3,warmup=500):
    if torch.cuda.device_count()>1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model=nn.DataParallel(model)
    model.to(device); optimizer=optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),weight_decay=0.01)
    scheduler=optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,total_steps=epochs*len(train_loader),pct_start=warmup/(epochs*len(train_loader)),anneal_strategy='cos')
    history={'loss':[],'acc':[],'time':[]}
    for ep in range(epochs):
        model.train(); total_loss=0.; total_correct=0; total_tokens=0; t0=time.time()
        for step,(inputs,targets) in enumerate(train_loader):
            inputs,targets=inputs.to(device),targets.to(device)
            logits=model(inputs)
            loss=nn.CrossEntropyLoss(ignore_index=pad_token)(logits.view(-1,vocab_size),targets.view(-1))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
            total_loss+=loss.item()*inputs.size(0)
            preds=logits.argmax(dim=-1); mask=(targets!=pad_token)
            correct=((preds==targets)&mask).sum().item(); total_tokens+=mask.sum().item(); total_correct+=correct
            if step%200==0: print(f"Ep {ep+1} Step {step:5d} | Loss: {loss.item():.4f} | Token Acc: {correct/mask.sum().item() if mask.sum()>0 else 0:.4f}")
        avg_loss=total_loss/len(train_loader.dataset); avg_acc=total_correct/total_tokens
        history['loss'].append(avg_loss); history['acc'].append(avg_acc); history['time'].append(time.time()-t0)
        print(f"=== Epoch {ep+1:2d} | Loss: {avg_loss:.4f} | Token Acc: {avg_acc:.4f} | Time: {time.time()-t0:.1f}s")
        if (ep+1)%5==0: torch.save(model.state_dict(),f'mbmd15_ep{ep+1}.pt')
    return history

train_dataset=NeuroSymbolicArithmeticDataset(n_samples=200000,max_digits=4,tool_threshold=3,seed=42)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,collate_fn=collate_fn,num_workers=2)
model=MiniGPT(vocab_size,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,max_len=256)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
history=train(model,train_loader,epochs=25,lr=1e-3,warmup=500)
torch.save(model.state_dict(),'mbmd15_final.pt')

def evaluate_neuro_symbolic(num_samples=200):
    model.eval()
    test_set=NeuroSymbolicArithmeticDataset(n_samples=num_samples,max_digits=4,tool_threshold=3,seed=999)
    correct_result=0; total=0
    for i,(inp_str,_) in enumerate(test_set.data):
        prompt=inp_str; generated=""; current=prompt
        with torch.no_grad():
            for _ in range(256):
                seq_ids=torch.tensor([encode_sequence(current,256)[:-1]],device=device)
                logits=model(seq_ids); next_token=torch.argmax(torch.softmax(logits[0,-1,:],dim=-1)).item()
                if next_token==eos_token: break
                token_str=itos[next_token]; generated+=token_str; current+=token_str
                if '</tool>' in generated:
                    match=re.search(r'<python>(.*?)</python>',generated)
                    if match:
                        code=match.group(1).strip()
                        try:
                            exec_result=eval(code)
                            current=current.rsplit('<tool>',1)[0]+f"<result>{exec_result}</result>"
                            generated=""
                        except: pass
                    break
        full_pred=current
        if '<result>' in full_pred:
            start=full_pred.rfind('<result>')+len('<result>'); end=full_pred.find('</result>',start)
            if end!=-1: pred_result=full_pred[start:end].strip()
        total+=1
    print(f"Neuro-Symbolic Exact Result Accuracy: {correct_result/total:.4f}")

evaluate_neuro_symbolic(num_samples=200)
