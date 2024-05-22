def get_wps():
    from dipmark import (
        Dip_Reweight,
        Soft_Reweight,
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
    )

    import random

    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")
#     delta_wp = WatermarkLogitsProcessor(
#         private_key,
#         Delta_Reweight(),
#         PrevN_ContextCodeExtractor(5),
#     )
#     gamma_wp = WatermarkLogitsProcessor(
#         private_key,
#         Gamma_Reweight(),
#         PrevN_ContextCodeExtractor(5),
#     )
#     deltagumbel_wp = WatermarkLogitsProcessor(
#         private_key,
#         DeltaGumbel_Reweight(),
#         PrevN_ContextCodeExtractor(5),
#     )
    
    dip_wps = [
        WatermarkLogitsProcessor(
        private_key,
        Dip_Reweight(alpha=alpha),
        PrevN_ContextCodeExtractor(5),
        )
        for alpha in [0.3, 0.35, 0.4, 0.45, 0.5]
    ]
#     import copy
#     delta_wp_woh = copy.deepcopy(delta_wp)
#     delta_wp_woh.ignore_history = True
#     gamma_wp_woh = copy.deepcopy(gamma_wp)
#     gamma_wp_woh.ignore_history = True
#     dip_wps_woh = [copy.deepcopy(dip_wps[2]),copy.deepcopy(dip_wps[3])]
#     dip_wps_woh[0].ignore_history = True
#     dip_wps_woh[1].ignore_history = True
    john_wps = [
        WatermarkLogitsProcessor(
        private_key,
        Soft_Reweight(delta=delta),
        PrevN_ContextCodeExtractor(5),
        ignore_history = True
        )
        for delta in [0.0, 1.0, 1.5, 2.0]
    ]

    return [
        None,
#         delta_wp,
#         gamma_wp,
#         delta_wp_woh,
#         gamma_wp_woh,
        *john_wps,
#         deltagumbel_wp,
        *dip_wps,
#         *dip_wps_woh
    ]



def get_num_gpus():
    import torch

    num_gpus = torch.cuda.device_count()
    return num_gpus


def batched_wp_task_worker(tq, get_in_ds, batch_size=8):
    ds = get_in_ds()

    from supplement_experiments.common import get_wps

    wps = get_wps()

    from tqdm import tqdm

    for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
        for wp in wps:
            tq.put({"batch": batch, "watermark_processor": wp})


def merged_task_worker(
    get_in_ds,
    output_filepath,
    tq,
    batch_size=8,
    watermark_only=False,
    wh_only=False,
    no_gumbel=False,
):
    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": output_filepath})["test"]
    out_ds = out_ds.sort("id")

    dss, wps = add_reference(in_ds, out_ds)

    from tqdm import tqdm

    for ds, wp_str in zip(dss, wps):
        if watermark_only:
            if "John" in wp_str or "None" == wp_str:
                continue
        if wh_only:
            if ", True)" in wp_str:
                continue
        if no_gumbel:
            if "Gumbel" in wp_str:
                continue
        for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
            tq.put(batch)


def log(line: dict, f):
    import json

    f.write(json.dumps(line))
    f.write("\n")
    f.flush()


def simple_store_worker(path, rq, rqe):
    import os

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    from queue import Empty

    with open(path, "w") as f:
        while not (rqe.is_set() and rq.empty()):
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            assert isinstance(result, dict)
            if result == {}:
                continue
            if isinstance(next(iter(result.values())), list):
                assert all([isinstance(v, list) for v in result.values()])
                lens = [len(v) for v in result.values()]
                assert all([l == lens[0] for l in lens])
                for i in range(lens[0]):
                    log({k: v[i] for k, v in result.items()}, f)
            else:
                log(result, f)


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}


def tokenize_batch(
    example,
    tokenizer,
    fields=["input"],
    max_length: int | dict = 512,
    task_template: str | dict = {},
    padding_side={},
):
    if isinstance(task_template, str):
        #  like "{input}"
        task_template = {"input": task_template}
    result = {}

    if tokenizer.name_or_path == "facebook/mbart-large-en-ro":
        tokenizer.tgt_lang = "ro_RO"
    if tokenizer.name_or_path == "daryl149/llama-2-7b-chat-hf":
        tokenizer.pad_token = tokenizer.eos_token

    for field in fields:
        if field in example:
            kwargs = {}
            if isinstance(max_length, dict):
                kwargs["max_length"] = max_length[field]
            else:
                kwargs["max_length"] = max_length
            if field in task_template:
                texts = [
                    task_template[field].format(**{field: s}) for s in example[field]
                ]
            else:
                texts = example[field]
            if field in ["output", "reference"]:
                kwargs["text_target"] = texts
            else:
                kwargs["text"] = texts
            if field == "output":
                kwargs["add_special_tokens"] = False
            if field in padding_side:
                kwargs["padding_side"] = padding_side[field]
            result[field] = tokenizer(
                **kwargs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

    return result


def set_spawn():
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def remove_tailing_pad_s(s: str):
    while s.endswith("<pad>"):
        s = s[:-5]
    return s
    #  index = s.find("<pad>")
    #  if index == -1:
    #      return s
    #  else:
    #      return s[:index]


def remove_tailing_pad(strs: list[str]):
    return [remove_tailing_pad_s(s) for s in strs]


def transformer_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    model_str,
    generation_kwargs={},
    decoder_only=False,
    tokenization_kwargs={},
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import AutoModelForCausalLM
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from dipmark import patch_model

    if decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            task = tq.get(timeout=1)
        except Empty as e:
            continue
        batch = task["batch"]
#         print(batch)
#         xDDDD
        tbatch = tokenize_batch(batch, tokenizer, **tokenization_kwargs)
        wp = task["watermark_processor"]
        lps = []
        if wp is not None:
            if "reset_history" in dir(wp):
                wp.reset_history()
            if "vocab_size" in dir(wp):
                wp.vocab_size = model.config.vocab_size
            lps.append(wp)

        # for reproducibility and sufficient randomness
        import hashlib

        hash = hashlib.sha256()
        hash.update(str(batch["id"]).encode("utf-8"))
        seed = hash.digest()
        seed = int.from_bytes(seed, "big") % (2**32 - 1)

        set_seed(seed)
#         print(tbatch["input"])
        
        outputs_ids = model.generate(
            tbatch["input"]["input_ids"].to(device=model.device),
            attention_mask=tbatch["input"]["attention_mask"].to(device=model.device),
            do_sample=True,
            num_beams=1,
            #  top_k=50,   # default
            logits_warper=LogitsProcessorList(lps),
            **generation_kwargs,
        )
        
        if decoder_only:
            outputs_ids = outputs_ids[:, tbatch["input"]["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=False)
        outputs = remove_tailing_pad(outputs)
        display_outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        wp_str = repr(wp)
        rq.put(
            {
                "output": outputs,
                "prompt" : batch["input"],
                "display_output": display_outputs,
                "id": batch["id"],
                "watermark_processor": [wp_str] * len(outputs),
            }
        )


def add_reference(in_ds, out_ds):
    """assuming ordered by ids"""
    wp_types = set(out_ds["watermark_processor"])

    s_out_dss = []
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        s_out_ds = s_out_ds.add_column("input", in_ds["input"])
        s_out_ds = s_out_ds.add_column("reference", in_ds["reference"])
        s_out_dss.append(s_out_ds)
    from datasets import concatenate_datasets

    return s_out_dss, wp_types


def bertscore_worker(tq, tqe, rq, gpu_id=0):
    import bert_score

    scorer = bert_score.BERTScorer(
        lang="de",
        rescale_with_baseline=True,
        device=f"cuda:{gpu_id}",
        use_fast_tokenizer=True,
    )

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        (P, R, F) = scorer.score(batch["display_output"], batch["reference"])
        rq.put(
            {
                **batch,
                "bertscore.precision": P.tolist(),
                "bertscore.recall": R.tolist(),
                "bertscore.f1": F.tolist(),
            }
        )


def rouge_worker(tq, tqe, rq):
    import evaluate

    rouge = evaluate.load("rouge")

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        rouge_scores = rouge.compute(
            predictions=batch["display_output"],
            references=batch["reference"],
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
            use_aggregator=False,
        )
        rq.put({**rouge_scores, **batch})


def remove_text_worker(tq, tqe, rq):
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        for f in ["input", "output", "reference", "display_output"]:
            if f in batch:
                del batch[f]
        rq.put(batch)


import torch


@torch.no_grad()
def get_ppl(model, tbatch, decoder_only=False):
    if not decoder_only:
        input_ids = tbatch["input"]["input_ids"].to(model.device)
        attention_mask = tbatch["input"]["attention_mask"].to(model.device)
        decoder_input_ids = tbatch["output"]["input_ids"][..., :-1].to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        #  labels: [batch_size, sequence_length]
        labels = tbatch["output"]["input_ids"][..., 1:].to(model.device)
        label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(
            model.device
        )
        #  output.logits: [batch_size, sequence_length, vocab_size]
        logits = outputs.logits
    else:
        com_input_ids = torch.cat(
            [tbatch["input"]["input_ids"], tbatch["output"]["input_ids"][..., :-1]],
            dim=-1,
        ).to(model.device)
        com_attention_mask = torch.cat(
            [
                tbatch["input"]["attention_mask"],
                tbatch["output"]["attention_mask"][..., :-1],
            ],
            dim=-1,
        ).to(model.device)
        outputs = model(input_ids=com_input_ids, attention_mask=com_attention_mask)
        labels = tbatch["output"]["input_ids"].to(model.device)
        label_attention_mask = tbatch["output"]["attention_mask"].to(model.device)
        logits = outputs.logits[:, tbatch["input"]["input_ids"].shape[1] - 1 :]

    from torch.nn import CrossEntropyLoss

    loss_fct = CrossEntropyLoss(reduction="none")

    shape = labels.shape
    #  loss: [batch_size, sequence_length]
    losses = loss_fct(
        logits.reshape(-1, logits.shape[-1]),
        labels.view(-1),
    ).reshape(shape)
    #  loss: [batch_size]
    losses = (losses * label_attention_mask.float()).sum(
        dim=-1
    ) / label_attention_mask.sum(dim=-1)
    ppl = torch.exp(losses).cpu().tolist()
    return ppl


def ppl_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    oracle_model_str,
    decoder_only=False,
    tokenization_kwargs={},
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import AutoModelForCausalLM
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    if decoder_only:
        model = AutoModelForCausalLM.from_pretrained(oracle_model_str).to(
            f"cuda:{gpu_id}"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(oracle_model_str).to(
            f"cuda:{gpu_id}"
        )
    tokenizer = AutoTokenizer.from_pretrained(oracle_model_str)

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
            **tokenization_kwargs,
        )
        ppl = get_ppl(model, tbatch, decoder_only=decoder_only)
        with torch.cuda.device(model.device):
            torch.cuda.empty_cache()

        rq.put(
            {
                **batch,
                "ppl": ppl,
            }
        )


@torch.no_grad()
def get_score(model, tbatch, wp, scorer, test_config={}, la_wp=None):
    input_ids = tbatch["input"]["input_ids"].to(model.device)
    attention_mask = tbatch["input"]["attention_mask"].to(model.device)
    #  labels : [batch_size, output_sequence_length-1]
    labels = tbatch["output"]["input_ids"][..., 1:].to(model.device)
    label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(model.device)
    #  decoder_input_ids : [batch_size, output_sequence_length-1]
    decoder_input_ids = tbatch["output"]["input_ids"][..., :-1].to(model.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    from transformers import GenerationConfig

    generation_config = GenerationConfig.from_model_config(model.config)
    if "temperature" in test_config:
        generation_config.temperature = test_config["temperature"]
    if "top_k" in test_config:
        generation_config.top_k = test_config["top_k"]
    logits_processor = model._get_logits_processor(
        generation_config,
        input_ids_seq_length=labels.shape[-1],
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
    )
    logits_warper = model._get_logits_warper(generation_config)

    #  logits: [batch_size, output_sequence_length-1, vocab_size]
    logits = outputs.logits
    del outputs
    del input_ids
    del attention_mask
    with torch.cuda.device(model.device):
        torch.cuda.empty_cache()
    old_logits = torch.clone(logits)
    scores = torch.zeros(
        logits.shape[:-1] + (scorer.query_size(),), device=logits.device
    )
    if la_wp is not None:
        la_scores = torch.zeros(logits.shape[:-1], device=logits.device)
    else:
        la_scores = None
    for i in range(logits.size(1)):
        pre = decoder_input_ids[:, : i + 1]
        t = logits[:, i]
        t = logits_processor(pre, t)
        t = logits_warper(pre, t)
        old_logits[:, i] = t
        new_logits = wp(pre, t)
        scores[:, i] = wp.get_score(labels[:, i], old_logits[:, i], new_logits, scorer)
        if la_wp is not None:
            la_scores[:, i] = la_wp.get_la_score(pre, labels[:, i], logits.shape[-1])
    del logits

    # compute entropy
    import torch.nn.functional as F

    _logits = F.log_softmax(old_logits, dim=-1)
    _logits.nan_to_num_()
    entropy = -(torch.exp(_logits) * _logits).sum(dim=-1)

    scores = scores * label_attention_mask.unsqueeze(-1)
    if la_wp is not None:
        la_scores = la_scores * label_attention_mask
    entropy = entropy * label_attention_mask
    return scores, entropy, label_attention_mask, la_scores




## DIPMARK
@torch.no_grad()
def get_dip_score(vocab_size, tbatch, wp, device, test_config={}, la_wp=None):
    input_ids = tbatch["input"]["input_ids"].to(device)
    attention_mask = tbatch["input"]["attention_mask"].to(device)
    #  labels : [batch_size, output_sequence_length-1]
    labels = tbatch["output"]["input_ids"][..., 1:].to(device)
    label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(device)
#     label_attention_mask = 1-label_attention_mask
    
    #  decoder_input_ids : [batch_size, output_sequence_length-1]
    decoder_input_ids = tbatch["output"]["input_ids"][..., :-1].to(device)
#     print(decoder_input_ids.shape)
    scores = torch.zeros(
        decoder_input_ids.shape,device = device)
#     print(decoder_input_ids[0],label_attention_mask[0])
#     XDDDD
    for i in range(decoder_input_ids.size(1)-1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i+1]
        out = wp.get_green_token_quantile(pre,vocab_size,cur_token)
        scores[:, i] = torch.stack(out).reshape(-1)
#     import torch.nn.functional as F
#     scores = scores * label_attention_mask
    return scores, label_attention_mask


def dip_score_worker(tq, tqe, rq, gpu_id, model_str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    vocab_size = max(tokenizer.vocab.values())
#     print(vocab_size)
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        assert len(set(batch["watermark_processor"])) == 1

        from unbiased_watermark import (
            Gamma_Reweight,
            Dip_Reweight,
            Delta_Reweight,
            DeltaGumbel_Reweight,
            Soft_Reweight,
            WatermarkLogitsProcessor,
            PrevN_ContextCodeExtractor,
        )
        
        wp_str = batch["watermark_processor"][0]
#         print(wp_str)
        wp = eval(wp_str)
        wp.ignore_history = True

        la_wp = None
#         print("start tokenizer")
        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
        )
        # score: [batch_size, sequence_length, query_size]
        # entropy: [batch_size, sequence_length]
        # label_attention_mask: [batch_size, sequence_length]
#         for i in range(500):
        score, label_attention_mask = get_dip_score(
            vocab_size, tbatch, wp, device, la_wp=la_wp
        )
#         print(label_attention_mask[0])
        score = score * label_attention_mask
        gamma_list  = [0.3,0.4,0.5,0.6,0.7]
        score_col = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        prob_col = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        num_g_tokens = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        seq_len = torch.sum(label_attention_mask,dim=-1,keepdim=False)
        for i,gm in enumerate(gamma_list):
            
#             print((1-gm)*seq_len)
            green_tokens = torch.sum(score>=gm,dim=-1,keepdim=False)
#             print(green_tokens)
            mid2=(green_tokens - (1-gm)*seq_len)/torch.sqrt(seq_len)
            mid=green_tokens/seq_len
            kl_div = mid*torch.log(mid/(1-gm))+(1-mid)*torch.log((1-mid)/gm)
            prob = torch.exp(-kl_div*seq_len)
            score_col[:,i] = torch.exp(-2*mid2*mid2)
            prob_col[:,i] = prob
            num_g_tokens[:,i] = green_tokens
            
        best_app_score = torch.min(score_col,dim = -1).values.cpu().tolist()
        best_score = torch.min(prob_col,dim = -1).values.cpu().tolist()
        john_score = prob_col[...,2].cpu().tolist()
        john_app_score = score_col[...,2].cpu().tolist()
        g_tokens = num_g_tokens[...,2].cpu().tolist()
        max_g_tokens = torch.max(num_g_tokens,dim = -1).values.cpu().tolist()
#         print(best_score)
#         cum_label_attention_mask = torch.cumsum(label_attention_mask, dim=-1)
#         lens = cum_label_attention_mask[:, -1]
        #  assert attention is like 11110000
#         print(lens)
#         _lens_m_1 = torch.argmax(
#             cum_label_attention_mask,
#             dim=-1,
#         )
#         print(_lens_m_1)
#         assert torch.all(lens == _lens_m_1 + 1)

#         lens = lens.cpu().tolist()

        rq.put(
            {
                **batch,
                "score": score.cpu().tolist(),
#                 "gamma_list": gamma_list,
                "best_score": best_score,
                "best_app_score":best_app_score,
                "john_score": john_score,
                "john_app_score": john_app_score,
                "g_tokens": g_tokens,
                "max_g_tokens": max_g_tokens,
                "lens": seq_len.cpu().tolist(),
                "all_scores": prob_col.cpu().tolist(),
                "app_scores": score_col.cpu().tolist(),
            }
        )

    
def get_robust_score(vocab_size, tbatch, wp, device, test_config={}, la_wp=None,eps=0.05):
    input_ids = tbatch["input"]["input_ids"].to(device)
    attention_mask = tbatch["input"]["attention_mask"].to(device)
    #  labels : [batch_size, output_sequence_length-1]
    labels = tbatch["output"]["input_ids"][..., 1:].to(device)
    label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(device)
#     label_attention_mask = 1-label_attention_mask
    
    #  decoder_input_ids : [batch_size, output_sequence_length-1]
    decoder_input_ids = tbatch["output"]["input_ids"][..., :-1].to(device)
    decoder_input_ids_noise = torch.rand_like(decoder_input_ids,dtype = torch.float32)
#     eps = 0.05
#     decoder_input_ids_noise = decoder_input_ids_noise<eps
    indices = torch.where(decoder_input_ids_noise<eps)#decoder_input_ids_noise.nonzero()
#     print(indices.shape)
#     xDDD
#     print(decoder_input_ids.dtype)
    noise = torch.randint_like(decoder_input_ids,0,vocab_size).to(torch.int64).to(device)
    decoder_input_ids[indices] = noise[indices]
#     print(decoder_input_ids[indices])
    
#     print(decoder_input_ids.shape)
    scores = torch.zeros(decoder_input_ids.shape,device = device)
#     print(decoder_input_ids[0],label_attention_mask[0])
#     XDDDD
    for i in range(decoder_input_ids.size(1)-1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_green_token_quantile(pre,vocab_size,cur_token)
        scores[:, i] = torch.stack(out).reshape(-1)
#     import torch.nn.functional as F
#     scores = scores * label_attention_mask
    return scores, label_attention_mask

def robust_score_worker(tq, tqe, rq, gpu_id, model_str, eps):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    vocab_size = max(tokenizer.vocab.values())
#     print(vocab_size)
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        assert len(set(batch["watermark_processor"])) == 1

        from unbiased_watermark import (
            Gamma_Reweight,
            Dip_Reweight,
            Delta_Reweight,
            DeltaGumbel_Reweight,
            Soft_Reweight,
            WatermarkLogitsProcessor,
            PrevN_ContextCodeExtractor,
        )
        
        wp_str = batch["watermark_processor"][0]
#         print(wp_str)
        wp = eval(wp_str)
        wp.ignore_history = True

        la_wp = None
#         print("start tokenizer")
        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
        )
        # score: [batch_size, sequence_length, query_size]
        # entropy: [batch_size, sequence_length]
        # label_attention_mask: [batch_size, sequence_length]
        score, label_attention_mask = get_robust_score(
            vocab_size, tbatch, wp, device, la_wp=la_wp, eps=eps
        )
#         print(label_attention_mask[0])
        score = score * label_attention_mask
        gamma_list  = [0.3,0.4,0.5,0.6,0.7]
        score_col = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        prob_col = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        num_g_tokens = torch.zeros([score.shape[0]]+[len(gamma_list)],device = device)
        seq_len = torch.sum(label_attention_mask,dim=-1,keepdim=False)
        for i,gm in enumerate(gamma_list):
            
#             print((1-gm)*seq_len)
            green_tokens = torch.sum(score>=gm,dim=-1,keepdim=False)
#             print(green_tokens)
            mid2=(green_tokens - (1-gm)*seq_len)/torch.sqrt(seq_len)
            mid=green_tokens/seq_len
            kl_div = mid*torch.log(mid/(1-gm))+(1-mid)*torch.log((1-mid)/gm)
            prob = torch.exp(-kl_div*seq_len)
            score_col[:,i] = torch.exp(-2*mid2*mid2)
            prob_col[:,i] = prob
            num_g_tokens[:,i] = green_tokens
            
        best_app_score = torch.min(score_col,dim = -1).values.cpu().tolist()
        best_score = torch.min(prob_col,dim = -1).values.cpu().tolist()
        john_score = prob_col[...,2].cpu().tolist()
        john_app_score = score_col[...,2].cpu().tolist()
        g_tokens = num_g_tokens[...,2].cpu().tolist()
        max_g_tokens = torch.max(num_g_tokens,dim = -1).values.cpu().tolist()
#         print(best_score)
#         cum_label_attention_mask = torch.cumsum(label_attention_mask, dim=-1)
#         lens = cum_label_attention_mask[:, -1]
        #  assert attention is like 11110000
#         print(lens)
#         _lens_m_1 = torch.argmax(
#             cum_label_attention_mask,
#             dim=-1,
#         )
#         print(_lens_m_1)
#         assert torch.all(lens == _lens_m_1 + 1)

#         lens = lens.cpu().tolist()

        rq.put(
            {
                **batch,
                "score": score.cpu().tolist(),
#                 "gamma_list": gamma_list,
                "best_score": best_score,
                "best_app_score":best_app_score,
                "john_score": john_score,
                "john_app_score": john_app_score,
                "g_tokens": g_tokens,
                "max_g_tokens": max_g_tokens,
                "lens": seq_len.cpu().tolist(),
                "all_scores": prob_col.cpu().tolist(),
                "app_scores": score_col.cpu().tolist(),
            }
        )
                