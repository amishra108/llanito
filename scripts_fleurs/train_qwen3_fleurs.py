from __future__ import annotations

import argparse, csv, io, json, math, os, random, time, unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, DatasetDict, load_from_disk
from opencc import OpenCC
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, StoppingCriteria, StoppingCriteriaList, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
SR = 16000
MAX_SAMPLES = 30 * SR
SEED = 2026
LORA_RANK = 128
LORA_ALPHA = 256.0
LORA_DROPOUT = 0.05
EXPECTED_TARGETS = 208
FULL_LANGS = {"cmn_hans_cn", "es_419", "fa_ir"}
QWEN_LANG = {
    "ast_es": "Spanish", "cmn_hans_cn": "Chinese", "es_419": "Spanish",
    "fa_ir": "Persian", "ky_kg": "Turkish", "ckb_iq": "Persian",
    "en_us": "English", "nl_nl": "Dutch", "pl_pl": "Polish", "th_th": "Thai",
}
ZH_T2S = OpenCC("t2s")


def p(*a): print(*a, flush=True)

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def data_root(lang):
    from scripts_fleurs.fleurs_paths import dataset_root
    return dataset_root(lang)

def result_dir(lang, method, trial):
    sub = (
        "fft"
        if method == "fft"
        else f"lora_r{LORA_RANK}"
    )
    return ROOT / "outputs" / "new_results_qwen" / lang / sub / f"trial_{trial:04d}"

def pcm_float(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating): return x.astype(np.float32, copy=False)
    if x.dtype == np.uint8: return ((x.astype(np.float32)-128.0)/128.0).astype(np.float32)
    info = np.iinfo(x.dtype); return (x.astype(np.float32)/float(max(abs(info.min), abs(info.max)))).astype(np.float32)

def decode_audio(v):
    if isinstance(v, dict):
        if v.get("array") is not None: x, sr = np.asarray(v["array"]), int(v.get("sampling_rate") or SR)
        elif v.get("bytes") is not None: x, sr = sf.read(io.BytesIO(v["bytes"]), always_2d=False)
        elif v.get("path"): x, sr = sf.read(str(v["path"]), always_2d=False)
        else: raise RuntimeError("Undecodable audio dict")
    elif isinstance(v, (str, Path)): x, sr = sf.read(str(v), always_2d=False)
    elif hasattr(v, "get_all_samples"):
        s = v.get_all_samples(); x = s.data.detach().cpu().numpy() if torch.is_tensor(s.data) else s.data; sr = int(s.sample_rate)
    else: raise TypeError(type(v))
    x = pcm_float(x)
    if x.ndim == 2: x = x.mean(axis=1)
    if x.ndim != 1: raise RuntimeError(f"bad audio shape {x.shape}")
    if sr != SR:
        g = math.gcd(int(sr), SR); x = resample_poly(x, SR//g, int(sr)//g).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if not len(x): raise RuntimeError("empty audio")
    return np.ascontiguousarray(x)

def load_split(lang, split):
    root = data_root(lang)
    if (root / "dataset_dict.json").exists():
        obj = load_from_disk(str(root)); ds = obj[split]
    else:
        ds = load_from_disk(str(root / split))
    if not {"audio", "transcription"}.issubset(ds.column_names): raise RuntimeError(f"bad columns: {ds.column_names}")
    raw = len(ds)
    if "num_samples" in ds.column_names:
        keep = [i for i,n in enumerate(ds["num_samples"]) if int(n) <= MAX_SAMPLES]
    else:
        keep = [i for i in range(raw) if len(decode_audio(ds[i]["audio"])) <= MAX_SAMPLES]
    ds = ds.select(keep)
    p(f"[duration-filter] {split}: raw={raw} kept={len(ds)} excluded_over_30s={raw-len(ds)}")
    try: ds = ds.cast_column("audio", Audio(decode=False))
    except Exception: pass
    return ds

def norm_text(text):
    out=[]
    for ch in unicodedata.normalize("NFKC", str(text)).casefold():
        try: out.append(str(int(unicodedata.digit(ch)))); continue
        except (TypeError, ValueError): pass
        cat = unicodedata.category(ch)
        out.append(" " if cat[0] in {"P","S"} or cat in {"Cf","Cc"} else ch)
    return " ".join("".join(out).split())

def edit_counts(r,h):
    n,m=len(r),len(h); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(r[i-1]!=h[j-1]))
    # counts via a second DP with deterministic S,D,I preference
    i,j=n,m; s=d=ins=0
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (r[i-1]!=h[j-1]):
            s += int(r[i-1]!=h[j-1]); i-=1; j-=1
        elif i and dp[i][j] == dp[i-1][j] + 1: d+=1; i-=1
        else: ins+=1; j-=1
    return s,d,ins

def metric(refs, hyps, lang):
    s=d=ins=n=0
    for ref,hyp in zip(refs,hyps,strict=True):
        rn,hn=norm_text(ref),norm_text(hyp)
        if lang == "cmn_hans_cn":
            rn,hn=ZH_T2S.convert(rn),ZH_T2S.convert(hn); rt=[c for c in rn if not c.isspace()]; ht=[c for c in hn if not c.isspace()]
        else: rt,ht=rn.split(),hn.split()
        ss,dd,ii=edit_counts(rt,ht); s+=ss; d+=dd; ins+=ii; n+=len(rt)
    return {"normalised_wer":100.0*(s+d+ins)/n if n else 0.0,"normalised_substitutions":s,"normalised_deletions":d,"normalised_insertions":ins,"normalised_reference_words":n,"metric":"CER" if lang=="cmn_hans_cn" else "WER"}

class CycleGuard(StoppingCriteria):
    def __init__(self,prompt_length,max_cycle_length=8,repeats_required=6): super().__init__(); self.p=prompt_length; self.m=max_cycle_length; self.r=repeats_required
    def __call__(self,input_ids,scores,**kwargs):
        g=input_ids[:,self.p:]; stop=torch.zeros(g.shape[0],dtype=torch.bool,device=g.device)
        for k in range(1,self.m+1):
            need=k*self.r
            if g.shape[1] >= need:
                tail=g[:,-need:]; stop |= (tail == tail[:,-k:].repeat(1,self.r)).all(dim=1)
        return stop

def prefix_text(processor):
    msgs=[[{"role":"system","content":""},{"role":"user","content":[{"type":"audio","audio":None}]}]]
    x=processor.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
    return x[0] if isinstance(x,(list,tuple)) else x

def collate(processor, qlang, rows):
    pfx=prefix_text(processor); aud=[decode_audio(r["audio"]) for r in rows]
    if any(len(x)>MAX_SAMPLES for x in aud): raise RuntimeError("audio >30s reached collator")
    prefixes=[pfx]*len(rows); eos=processor.tokenizer.eos_token or ""
    targets=[f"language {qlang}<asr_text>{r['transcription']}" for r in rows]
    full=[a+b+eos for a,b in zip(prefixes,targets,strict=True)]
    fi=processor(text=full,audio=aud,return_tensors="pt",padding=True,truncation=False)
    pi=processor(text=prefixes,audio=aud,return_tensors="pt",padding=True,truncation=False)
    labels=fi["input_ids"].clone(); lens=pi["attention_mask"].sum(dim=1).tolist()
    for i,L in enumerate(lens): labels[i,:int(L)] = -100
    pad=processor.tokenizer.pad_token_id
    if pad is not None: labels[labels==int(pad)] = -100
    fi["labels"] = labels; return dict(fi)

def loader(ds,processor,qlang,batch,shuffle,epoch=0):
    gen=torch.Generator().manual_seed(SEED+epoch) if shuffle else None
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,generator=gen,num_workers=0,pin_memory=True,collate_fn=lambda rows: collate(processor,qlang,rows))

def move(batch,device,dtype):
    out={}
    for k,v in batch.items():
        if torch.is_tensor(v):
            v=v.to(device,non_blocking=True)
            if v.is_floating_point(): v=v.to(dtype=dtype)
        out[k]=v
    return out

def attention_targets(model):
    mods=dict(model.named_modules()); out={}
    for name,m in mods.items():
        if not isinstance(m,nn.Linear): continue
        if name.rsplit(".",1)[-1] not in {"q_proj","k_proj","v_proj","o_proj","out_proj"}: continue
        parent=mods.get(name.rsplit(".",1)[0])
        if parent is not None and "attention" in parent.__class__.__name__.casefold(): out[name]=m
    if len(out)!=EXPECTED_TARGETS: raise RuntimeError(f"expected {EXPECTED_TARGETS} attention targets, got {len(out)}")
    return out

def parent_of(model,name):
    parts=name.split("."); p=model
    for x in parts[:-1]: p=getattr(p,x)
    return p,parts[-1]

class LoRALinear(nn.Module):
    def __init__(self,base):
        super().__init__(); self.base=base; self.scale=LORA_ALPHA/LORA_RANK; self.drop=nn.Dropout(LORA_DROPOUT)
        for p in base.parameters(): p.requires_grad=False
        self.lora_A=nn.Parameter(torch.empty(LORA_RANK,base.in_features,dtype=torch.float32,device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,LORA_RANK,dtype=torch.float32,device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        y=self.base(x); z=F.linear(self.drop(x).float(),self.lora_A); z=F.linear(z,self.lora_B); return y+(self.scale*z).to(y.dtype)

def configure(model,method):
    if method=="fft":
        for p in model.parameters(): p.requires_grad=True
        return []
    targets=attention_targets(model)
    for p in model.parameters(): p.requires_grad=False
    for name in sorted(targets):
        par,leaf=parent_of(model,name); setattr(par,leaf,LoRALinear(getattr(par,leaf)))
    return sorted(targets)

@torch.inference_mode()
def transcribe(model,processor,audio,qlang,device,dtype):
    x=processor.apply_transcription_request(audio=audio,language=qlang); moved={}
    for k,v in dict(x).items():
        if torch.is_tensor(v):
            v=v.to(device); v=v.to(dtype=dtype) if v.is_floating_point() else v
        moved[k]=v
    ids=model.generate(**moved,max_new_tokens=512,do_sample=False,stopping_criteria=StoppingCriteriaList([CycleGuard(int(moved["input_ids"].shape[1]))]))
    gen=ids[:,int(moved["input_ids"].shape[1]):]; return str(processor.decode(gen,return_format="transcription_only")[0]).strip()

@torch.inference_mode()
def evaluate(model,processor,ds,lang,qlang,device,dtype,pred_path,eval_batch):
    model.eval(); model.config.use_cache=False
    total=tokens=batches=0
    for b in loader(ds,processor,qlang,eval_batch,False):
        b=move(b,device,dtype); count=int(b["labels"].ne(-100).sum().item())
        with torch.autocast("cuda",dtype=dtype): loss=model(**b).loss
        total += float(loss.item())*count; tokens += count; batches += 1
    refs=[]; hyps=[]; rows=[]; model.config.use_cache=True
    for i,row in enumerate(ds,start=1):
        ref=str(row["transcription"]); hyp=transcribe(model,processor,decode_audio(row["audio"]),qlang,device,dtype)
        refs.append(ref); hyps.append(hyp); rows.append({"index":i-1,"id":row.get("id",i-1),"reference_original":ref,"hypothesis_original":hyp,"reference_normalised":norm_text(ref),"hypothesis_normalised":norm_text(hyp)})
        if i%50==0 or i==len(ds): p(f"[validation decode] {i}/{len(ds)}")
    write_csv(pred_path,rows); m=metric(refs,hyps,lang); m.update({"validation_loss":total/tokens,"validation_supervised_tokens":tokens,"validation_loss_batches":batches}); model.config.use_cache=False; return m

def parse():
    a=argparse.ArgumentParser()
    a.add_argument("--language",required=True); a.add_argument("--qwen-language",default=None)
    a.add_argument("--method",choices=["fft","lora"],required=True); a.add_argument("--trial",type=int,required=True)
    a.add_argument("--learning-rate",type=float,required=True); a.add_argument("--effective-batch-size",type=int,required=True)
    a.add_argument("--warmup-ratio",type=float,default=0.05); a.add_argument("--microbatch",type=int,default=1); a.add_argument("--eval-batch",type=int,default=1)
    a.add_argument("--max-epochs",type=int,default=12); a.add_argument("--patience",type=int,default=5); a.add_argument("--weight-decay",type=float,default=0.01); a.add_argument("--max-grad-norm",type=float,default=1.0); a.add_argument("--device",default="cuda:0")
    a.add_argument("--lora-rank",type=int,default=128)
    a.add_argument("--lora-alpha",type=float,default=None)
    return a.parse_args()

def main():
    args=parse();

    global LORA_RANK
    global LORA_ALPHA

    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be > 0")

    LORA_RANK = int(args.lora_rank)
    LORA_ALPHA = (
        float(args.lora_alpha)
        if args.lora_alpha is not None
        else 2.0 * LORA_RANK
    )

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device(args.device); torch.cuda.set_device(device); set_seed()
    qlang=args.qwen_language or QWEN_LANG.get(args.language)
    if qlang is None: raise RuntimeError("unknown Qwen language mapping; pass --qwen-language")
    if args.effective_batch_size % args.microbatch: raise RuntimeError("EBS must divide by microbatch")
    ga=args.effective_batch_size//args.microbatch; out=result_dir(args.language,args.method,args.trial); complete=out/"COMPLETE"
    if complete.exists(): p(f"[skip] completed {out}"); return
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refusing non-empty incomplete output: {out}")
    out.mkdir(parents=True,exist_ok=True)
    train=load_split(args.language,"train"); val=load_split(args.language,"validation")
    processor=AutoProcessor.from_pretrained(BASE_MODEL); dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=Qwen3ASRForConditionalGeneration.from_pretrained(BASE_MODEL,dtype=dtype).to(device); model.config.use_cache=False
    targets=configure(model,args.method)
    if hasattr(model,"gradient_checkpointing_enable"):
        try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        except TypeError: model.gradient_checkpointing_enable()
    if args.method=="lora" and hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()
    params=[x for x in model.parameters() if x.requires_grad]
    probe=loader(train,processor,qlang,args.microbatch,True,1); updates=math.ceil(len(probe)/ga); total_steps=updates*args.max_epochs; warmup=int(round(total_steps*args.warmup_ratio))
    opt=torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=args.weight_decay); sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=warmup,num_training_steps=total_steps)
    cfg={"language":args.language,"qwen_language":qlang,"dataset_root":str(data_root(args.language)),"base_model":BASE_MODEL,"method":args.method,"trial":args.trial,"learning_rate":args.learning_rate,"effective_batch_size":args.effective_batch_size,"microbatch":args.microbatch,"gradient_accumulation_steps":ga,"warmup_ratio":args.warmup_ratio,"max_epochs":args.max_epochs,"patience":args.patience,"weight_decay":args.weight_decay,"max_grad_norm":args.max_grad_norm,"maximum_audio_seconds":30.0,"seed":SEED,"lora_rank":LORA_RANK if args.method=="lora" else None,"lora_alpha":LORA_ALPHA if args.method=="lora" else None,"lora_dropout":LORA_DROPOUT if args.method=="lora" else None,"target_module_count":len(targets),"train_examples":len(train),"validation_examples":len(val),"metric":"CER" if args.language=="cmn_hans_cn" else "WER","test_touched_during_training":False}
    save_json(out/"hyperparameters.json",cfg)
    p("="*90); p(f"QWEN3-ASR {args.method.upper()} {args.language} trial={args.trial:04d}"); p(f"dataset={data_root(args.language)} forced_language={qlang}"); p(f"lr={args.learning_rate} EBS={args.effective_batch_size} microbatch={args.microbatch} grad_acc={ga} warmup={args.warmup_ratio}"); p(f"train={len(train)} val={len(val)} metric={cfg['metric']}"); p("="*90)
    best=float("inf"); best_epoch=0; bad=0; history=[]; global_step=0; started=time.time()
    for epoch in range(1,args.max_epochs+1):
        model.train(); model.config.use_cache=False; opt.zero_grad(set_to_none=True); runloss=0.0; nb=0; tr=loader(train,processor,qlang,args.microbatch,True,epoch)
        for bi,b in enumerate(tr,start=1):
            b=move(b,device,dtype)
            with torch.autocast("cuda",dtype=dtype): loss=model(**b).loss
            if loss is None or not torch.isfinite(loss): raise RuntimeError(f"bad loss epoch={epoch} batch={bi}: {loss}")
            runloss += float(loss.detach().item()); nb += 1; start=((bi-1)//ga)*ga; group=min(ga,len(tr)-start); (loss/group).backward()
            if bi%ga==0 or bi==len(tr): torch.nn.utils.clip_grad_norm_(params,args.max_grad_norm); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_step += 1
            if bi%100==0 or bi==len(tr): p(f"[train] epoch={epoch} batch={bi}/{len(tr)} loss={runloss/nb:.6f}")
        curpred=out/"_current_validation_predictions.csv"; m=evaluate(model,processor,val,args.language,qlang,device,dtype,curpred,args.eval_batch); score=float(m["normalised_wer"]); improved=score < best-1e-12
        if improved: best=score; best_epoch=epoch; bad=0
        else: bad += 1
        row={"epoch":epoch,"global_step":global_step,"train_loss":runloss/max(nb,1),"validation_loss":m["validation_loss"],"validation_normalised_wer":score,"metric":m["metric"],"best_so_far":best,"best_epoch_so_far":best_epoch,"bad_epochs":bad,"learning_rate_after_epoch":float(opt.param_groups[0]["lr"]),"runtime_seconds_total":time.time()-started}; history.append(row); write_csv(out/"training_history.csv",history)
        if improved:
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}; payload={"schema":"qwen3_fft_lora_best_v1","base_model":BASE_MODEL,"language":args.language,"qwen_language":qlang,"method":args.method,"trial":args.trial,"epoch":epoch,"config":cfg,"state_dict":state}
            torch.save(payload,out/"best_checkpoint.pt"); os.replace(curpred,out/"validation_predictions.csv"); save_json(out/"validation_metrics.json",{**m,"best_epoch":best_epoch,"checkpoint":str(out/"best_checkpoint.pt")})
        else: curpred.unlink(missing_ok=True)
        p(f"[epoch] {epoch} train_loss={row['train_loss']:.6f} val_loss={row['validation_loss']:.6f} {m['metric']}={score:.4f}% best={best:.4f}%@{best_epoch} bad={bad}/{args.patience}")
        if args.patience>0 and bad>=args.patience: p(f"[early stopping] epoch={epoch} best={best:.4f}%@{best_epoch}"); break
    save_json(out/"trial_result.json",{"language":args.language,"method":args.method,"trial":args.trial,"best_validation_normalised_wer":best,"metric":"CER" if args.language=="cmn_hans_cn" else "WER","best_epoch":best_epoch,"epochs_ran":len(history),"checkpoint":str(out/"best_checkpoint.pt"),"test_has_not_been_run":True}); complete.write_text("complete\n",encoding="utf-8"); p(f"[COMPLETE] {out}")

if __name__ == "__main__": main()
