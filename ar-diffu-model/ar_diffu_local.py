"""Shared local runner for the Colab_AR_vs_*_Full notebooks.

Same prompts, generation kwargs, NLI judge, and output JSON schema/filenames as
the notebooks, adapted for local runs (CUDA / Apple MPS / CPU). No Ollama.
"""
import argparse
import gc
import inspect
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import CrossEncoder

QUESTIONS = [
    "What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the largest planet in our solar system?",
    "When was the Declaration of Independence signed?", "How many bones are in the human body?", "Who invented the telephone?",
    "What is the chemical formula for water?", "What is the speed of light?", "Who painted the Mona Lisa?",
    "What is the smallest country in the world?", "How many continents are there?", "What is the capital of Japan?",
    "Who was the first president of the United States?", "What is the boiling point of water?", "How long does it take Earth to orbit the Sun?",
    "What is the Pythagorean theorem?", "Who wrote 1984?", "What is the Great Wall of China?", "How many sides does a triangle have?",
    "What is photosynthesis?", "Who discovered gravity?", "What is the currency of Germany?", "What is the largest ocean on Earth?",
    "Who wrote the Bible?", "What is the population of China?", "What is DNA?", "Who invented the internet?",
    "What is the distance from Earth to the Sun?", "Who wrote Hamlet?", "What is the capital of Brazil?",
    "What is the highest mountain in the world?", "Who was Abraham Lincoln?", "What is the freezing point of water?",
    "What is the capital of Australia?", "Who invented the printing press?", "What is the periodic table?",
    "What is the capital of Egypt?", "Who was Cleopatra?", "What is the capital of India?", "What is the Great Barrier Reef?",
    "Who wrote Pride and Prejudice?", "What is the capital of Canada?", "What is the Statue of Liberty?",
    "Who invented the telephone?", "What is the capital of France?", "What is the largest desert in the world?",
    "Who wrote War and Peace?", "What is the capital of Spain?", "What is the Eiffel Tower?", "Who was Julius Caesar?",
    "What is the capital of Russia?", "What is the Amazon Rainforest?", "Who wrote The Great Gatsby?",
    "What is the capital of Italy?", "What is the Colosseum?", "Who was Napoleon Bonaparte?", "What is the capital of Mexico?",
    "What is the Grand Canyon?", "Who wrote Moby Dick?", "What is the capital of Greece?", "What is the Acropolis?",
    "Who was Alexander the Great?", "What is the capital of Turkey?", "What is the Taj Mahal?", "Who wrote Don Quixote?",
    "What is the capital of South Korea?", "What is Mount Fuji?", "Who was Leonardo da Vinci?", "What is the capital of Thailand?",
    "What is Angkor Wat?", "Who wrote Crime and Punishment?", "What is the capital of Vietnam?", "What is Ha Long Bay?",
    "Who was Vincent van Gogh?", "What is the capital of Netherlands?", "What is Big Ben?", "Who wrote Anna Karenina?",
    "What is the capital of Belgium?", "What is the Sagrada Familia?", "Who was Pablo Picasso?", "What is the capital of Portugal?",
    "What is the Christ the Redeemer statue?", "Who wrote The Odyssey?", "What is the capital of Sweden?", "What is the Aurora Borealis?",
    "Who was Marie Curie?", "What is the capital of Norway?", "What are the Northern Lights?", "Who wrote Ulysses?",
    "What is the capital of Switzerland?", "What is the Matterhorn?", "Who was Albert Einstein?",
]

REASONING_QUESTIONS = [
    "If a train travels 60 miles in 2 hours, what is its average speed in miles per hour?",
    "Tom has 3 apples and gives 1 away. How many apples does he have left?",
    "If today is Monday, what day will it be in 3 days?",
    "If a shirt costs $20 and is on sale for 25% off, what is the sale price?",
    "How many minutes are there in 2.5 hours?",
    "If a car travels at 50 mph for 3 hours, how far does it go?",
    "What is 15 plus 27?",
    "What is 9 multiplied by 6?",
    "If you have 10 cookies and share them equally among 5 friends, how many cookies does each friend get?",
    "A rectangle has a length of 8 and a width of 3. What is its area?",
    "If it is 3 PM now, what time will it be in 5 hours?",
    "If a dozen eggs costs $3, how much does one egg cost?",
    "Sarah is older than Tom, and Tom is older than Mike. Who is the youngest?",
    "If there are 7 days in a week, how many days are there in 4 weeks?",
    "What is half of 100?",
    "If a book has 300 pages and you read 50 pages a day, how many days will it take to finish?",
    "What is the next number in the sequence: 2, 4, 6, 8?",
    "If a pizza is cut into 8 slices and you eat 3, how many slices are left?",
    "How many centimeters are there in 1 meter?",
    "If today is January 1st, what date is it in 10 days?",
    "What is 100 divided by 4?",
    "If a bag contains 5 red balls and 3 blue balls, how many balls are there in total?",
    "If you double 12, what do you get?",
    "A train leaves at 9 AM and arrives 4 hours later. What time does it arrive?",
    "If three friends split a $30 bill equally, how much does each pay?",
    "What is 7 squared?",
    "If a year has 12 months, how many months are in half a year?",
    "If Alice has twice as many books as Bob, and Bob has 4 books, how many books does Alice have?",
    "How many seconds are in 3 minutes?",
    "If it takes 1 hour to bake a cake and you start at 2 PM, when will it be done?",
]

_GSM8K_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k50.json")
GSM8K_QUESTIONS = ([x["question"] for x in json.load(open(_GSM8K_PATH))]
                   if os.path.exists(_GSM8K_PATH) else [])

QUESTION_SETS = {
    "factual": QUESTIONS,
    "reasoning": REASONING_QUESTIONS,
    "gsm8k": GSM8K_QUESTIONS,  # 50 seeded GSM8K test items (data/gsm8k50.json)
}

AR_MODELS = {
    "qwen25-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen25-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen25-3b": "Qwen/Qwen2.5-3B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def half_dtype(device, preferred):
    # fp16/bf16 matmuls are slow or unsupported on CPU; fall back to fp32 there.
    return torch.float32 if device == "cpu" else preferred


def free_memory(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_ar(hf_id, device):
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=half_dtype(device, torch.float16))
    model.config.pad_token_id = tok.eos_token_id
    model.to(device).eval()
    return model, tok


def generate_responses(model, tokenizer, temperature, questions, device, label="", max_new_tokens=40):
    """Standard causal-LM generation (identical kwargs to the notebooks).

    Anti-repetition kwargs (repetition_penalty/no_repeat_ngram_size) are only
    supported by HF's built-in generate(); some diffusion models ship a custom
    generate() that rejects unknown kwargs, so we retry without them.
    """
    results = []
    use_chat_template = getattr(tokenizer, "chat_template", None) is not None
    for i, q in enumerate(questions, 1):
        prompt = f"Q: {q}\nA: "
        try:
            if use_chat_template:
                # Instruct models need chat-template formatting to actually follow
                # instructions; raw "Q:/A:" completion prompts waste their tuning.
                chat_out = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                input_ids = chat_out["input_ids"].to(device)
                inputs = {"input_ids": input_ids, "attention_mask": chat_out["attention_mask"].to(device)}
                prompt_len = input_ids.shape[1]
            else:
                # CID diffusion models' generate() rejects token_type_ids; GPT-2 ignores them.
                inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
                prompt_len = None
            gen_kwargs = dict(
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            ) if temperature > 0 else dict(do_sample=False, pad_token_id=tokenizer.eos_token_id)
            # Chat-templated prompts vary in length (system/turn tokens), so bound
            # the *new* tokens rather than the total sequence length.
            # 40 new tokens is enough for the actual answer sentence (established
            # within the first ~10-15 tokens); cuts AR generation time ~1.5-1.9x
            # vs 80 with no observed quality loss (both cut off mid-ramble anyway).
            # MiniCPM5's custom generate() only accepts max_length (prompt+answer
            # combined, not new-tokens-only); max_new_tokens+8 total ~= max_new_tokens-ish
            # new tokens after the ~12-15 token "Q:...A:" prompt overhead -- same
            # ballpark as the other models' new-token budget.
            # NOTE: this 40-token default was tuned for terse factual-recall answers
            # ("The capital of France is Paris.") and silently truncates AR instruct
            # models mid chain-of-thought on word-problem/reasoning questions (they
            # "show work" by default and the numeric answer lands after the cutoff).
            # Pass a larger max_new_tokens for reasoning-style question sets.
            gen_kwargs.update({"max_new_tokens": max_new_tokens} if use_chat_template else {"max_length": max_new_tokens + 8})
            with torch.no_grad():
                if use_chat_template:
                    # Instruct models rarely loop like base LMs; repetition_penalty
                    # here instead demotes space-prefixed BPE tokens and glues
                    # words together (e.g. "ofFranceis"), so skip it for these.
                    outputs = model.generate(**inputs, **gen_kwargs)
                else:
                    try:
                        outputs = model.generate(
                            **inputs, **gen_kwargs, repetition_penalty=1.15, no_repeat_ngram_size=3,
                        )
                    except (TypeError, ValueError):
                        outputs = model.generate(**inputs, **gen_kwargs)
            if prompt_len is not None:
                response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
            else:
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            # Base models keep "answering" by hallucinating the next Q/A turn;
            # cut the response off at the first sign of a new turn.
            for marker in ("\nQ:", "\nQ", "\nA:"):
                if marker in response:
                    response = response.split(marker)[0].strip()
            results.append({"question": q, "response": response})
        except Exception as e:
            print(f"\n    ✗ generation failed for {q!r}: {e}", flush=True)
            results.append({"question": q, "response": ""})
        _progress(label, i, len(questions))
    return results


def generate_llada_responses(model, tokenizer, temperature, questions, device, label=""):
    """LLaDA diffusion generation (chat template + diffusion kwargs; returns only new tokens)."""
    results = []
    for i, q in enumerate(questions, 1):
        prompt = f"Q: {q}\nA: "
        try:
            # Newer tokenizers return a BatchEncoding (dict-like) here instead of
            # a raw tensor; extract input_ids explicitly so model.generate() gets a Tensor.
            chat_out = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = chat_out["input_ids"] if hasattr(chat_out, "__getitem__") and "input_ids" in chat_out else chat_out
            input_ids = input_ids.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs=input_ids,
                    eos_early_stop=True,
                    gen_length=150,
                    block_length=32,
                    threshold=0.5,
                    editing_threshold=0,
                    temperature=temperature,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            results.append({"question": q, "response": response})
        except Exception as e:
            print(f"\n    ✗ diffusion generation failed for {q!r}: {e}", flush=True)
            results.append({"question": q, "response": ""})
        _progress(label, i, len(questions))
    return results


def _lladamoe_add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    # float64 isn't supported on MPS; float32 is precise enough for sampling noise.
    noise_dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    logits = logits.to(noise_dtype)
    noise = torch.rand_like(logits, dtype=noise_dtype)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _lladamoe_get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def _lladamoe_diffusion_generate(model, prompt, steps, gen_length, block_length, temperature, mask_id):
    """Block-wise iterative unmasking loop, reimplemented from the model card's
    reference `generate()` (this checkpoint exposes forward()/logits only, no
    HF-compatible .generate())."""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start = prompt.shape[1] + num_block * block_length
        end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, start:end] == mask_id
        num_transfer_tokens = _lladamoe_get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x).logits
            logits_with_noise = _lladamoe_add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            p = torch.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x


def generate_lladamoe_responses(model, tokenizer, temperature, questions, device, label="",
                                 gen_length=40, block_length=20, steps=32, mask_id=156895):
    """LLaDA-MoE-7B-A1B diffusion generation via the manual block-wise unmasking
    loop from the model card (no built-in .generate() for this checkpoint)."""
    # keep steps-per-block constant (16) when the caller scales gen_length up,
    # so a larger token budget doesn't also silently cut unmasking quality.
    if gen_length != 40:
        block_length = 20
        num_blocks = max(1, -(-gen_length // block_length))  # ceil div
        gen_length = num_blocks * block_length
        steps = num_blocks * 16
    results = []
    for i, q in enumerate(questions, 1):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": q},
            ]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = torch.tensor(tokenizer(prompt_text)["input_ids"], device=device).unsqueeze(0)
            out = _lladamoe_diffusion_generate(
                model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=temperature, mask_id=mask_id,
            )
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append({"question": q, "response": response})
        except Exception as e:
            print(f"\n    ✗ diffusion generation failed for {q!r}: {e}", flush=True)
            results.append({"question": q, "response": ""})
        _progress(label, i, len(questions))
    return results


def generate_dream_responses(model, tokenizer, temperature, questions, device, label="",
                              max_new_tokens=40, steps=20):
    """Dream-v0-Instruct-7B diffusion generation via its custom diffusion_generate()
    (note: NOT the standard .generate() -- Dream only overrides diffusion_generate).
    steps is deliberately << the model's default of 512 for speed; max_new_tokens/steps
    tokens are unmasked per diffusion step, trading some quality for speed."""
    # scale steps proportionally so a larger token budget doesn't thin out the
    # per-token unmasking quality (keeps ~0.5 step/token, same ratio as the default).
    if max_new_tokens != 40:
        steps = max(20, round(max_new_tokens * 0.5))
    results = []
    for i, q in enumerate(questions, 1):
        try:
            messages = [{"role": "user", "content": q}]
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", return_dict=True, add_generation_prompt=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    steps=steps,
                    temperature=temperature,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.0,
                    return_dict_in_generate=True,
                )
            response = tokenizer.decode(
                output.sequences[0][input_ids.shape[1]:].tolist(), skip_special_tokens=True
            ).strip()
            results.append({"question": q, "response": response})
        except Exception as e:
            print(f"\n    ✗ diffusion generation failed for {q!r}: {e}", flush=True)
            results.append({"question": q, "response": ""})
        _progress(label, i, len(questions))
    return results


def _progress(label, i, n):
    if label and (i % 10 == 0 or i == n):
        print(f"    [{label}] {i}/{n}", flush=True)


def output_path(output_dir, ar_name, ar_temp, diffu_name, diffu_temp):
    return os.path.join(
        output_dir,
        f"ar-{ar_name}-{int(ar_temp * 100)}-diffu-{diffu_name}-{int(diffu_temp * 100)}.json",
    )


def compare_and_save(judge, ar_results, diffu_results, ar_name, ar_temp, diffu_name, diffu_temp, filename):
    comparison = []
    for ar_res, diffu_res in zip(ar_results, diffu_results):
        nli_score = float(judge.predict([[diffu_res["response"], ar_res["question"]]])[0])
        comparison.append({
            "question": ar_res["question"],
            "ar_response": ar_res["response"],
            "diffu_response": diffu_res["response"],
            "nli_score": nli_score,
            "hallucinated": nli_score < 0.5,
        })

    halluc_count = sum(1 for c in comparison if c["hallucinated"])
    truthful_pct = 100 * (len(comparison) - halluc_count) / len(comparison)

    tmp = filename + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "ar_model": ar_name,
            "ar_temperature": ar_temp,
            "diffu_model": diffu_name,
            "diffu_temperature": diffu_temp,
            "truthful_pct": truthful_pct,
            "hallucinated_count": halluc_count,
            "comparisons": comparison,
        }, f, indent=2)
    os.replace(tmp, filename)
    print(f"  ✓ {filename} ({truthful_pct:.0f}% truthful)", flush=True)


def build_parser(description, default_diffu_model, default_out_dir):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--diffu-model", default=default_diffu_model, help="HF model id for the diffusion model")
    p.add_argument("--out-dir", default=default_out_dir)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="Device for AR models + judge")
    p.add_argument("--diffu-device", default=None, choices=["cuda", "mps", "cpu"],
                   help="Device for the diffusion model (defaults to --device)")
    p.add_argument("--temps", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--ar-models", nargs="+", default=list(AR_MODELS), choices=list(AR_MODELS))
    p.add_argument("--limit", type=int, default=None, help="Only use first N questions (smoke test)")
    p.add_argument("--question-set", default="factual", choices=list(QUESTION_SETS),
                   help="Which question set to use: 'factual' (92 trivia Qs) or 'reasoning' (30 arithmetic/logic Qs)")
    p.add_argument("--gen-tokens", type=int, default=40,
                   help="Max new tokens per response for AR+diffusion models (default 40, tuned for terse "
                        "factual answers). Bump this for --question-set reasoning: AR instruct models "
                        "'show work' by default and get truncated before stating the final answer at 40.")
    p.add_argument("--overwrite", action="store_true", help="Re-run comparisons whose output file already exists")
    p.add_argument("--reuse-diffu", action="store_true",
                   help="Generate diffusion responses once per temperature and reuse across AR models "
                        "(~3x faster; notebooks regenerate them for every AR model)")
    p.add_argument("--seed", type=int, default=None)
    return p


def load_cached_ar(ar_name, question_set, gen_tokens, temp, questions):
    """Reuse AR outputs from run_ar_only.py when an exactly matching run exists."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ar_only",
                        f"{ar_name}-{question_set}-k{gen_tokens}-t{round(temp * 100)}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cached = json.load(f)["results"]
    if [r["question"] for r in cached] != list(questions):
        return None
    print(f"    ↺ reusing AR outputs from {os.path.relpath(path)}", flush=True)
    return cached


def run(args, diffu_name, load_diffu, diffu_generate):
    """Run AR × temps vs diffusion model; mirrors notebook Step 4 loop order."""
    device = pick_device(args.device)
    diffu_device = args.diffu_device or device
    questions = QUESTION_SETS[args.question_set][: args.limit] if args.limit else QUESTION_SETS[args.question_set]
    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    plan = [
        (ar_name, t, output_path(args.out_dir, ar_name, t, diffu_name, t))
        for ar_name in args.ar_models
        for t in args.temps
    ]
    todo = [p for p in plan if args.overwrite or not os.path.exists(p[2])]
    print(f"Device: {device} (diffusion: {diffu_device}) | questions: {len(questions)} | "
          f"comparisons: {len(todo)}/{len(plan)} to run -> {args.out_dir}/", flush=True)
    todo_files = {f for _, _, f in todo}
    for f in (f for _, _, f in plan if f not in todo_files):
        print(f"  ↷ skip (exists): {f}")
    if not todo:
        print("Nothing to do.")
        return

    print("Loading NLI judge (cross-encoder/qnli-distilroberta-base)...", flush=True)
    judge = CrossEncoder("cross-encoder/qnli-distilroberta-base", device=device)

    print(f"Loading diffusion model {args.diffu_model} on {diffu_device}...", flush=True)
    t0 = time.time()
    diff_model, diff_tok = load_diffu(args.diffu_model, diffu_device)
    print(f"✓ diffusion model loaded ({time.time() - t0:.0f}s)\n", flush=True)

    diffu_cache = {}
    start = time.time()
    done = 0
    for ar_name in args.ar_models:
        runs = [(t, f) for n, t, f in todo if n == ar_name]
        if not runs:
            continue
        print(f"\n{ar_name.upper()}:", flush=True)
        ar_model = ar_tok = None
        for temp, filename in runs:
            print(f"  temp={temp}...", flush=True)
            ar_results = load_cached_ar(ar_name, args.question_set, args.gen_tokens, temp, questions)
            if ar_results is None:
                if ar_model is None:
                    ar_model, ar_tok = load_ar(AR_MODELS[ar_name], device)
                ar_results = generate_responses(ar_model, ar_tok, temp, questions, device, label=f"{ar_name} AR",
                                                 max_new_tokens=args.gen_tokens)
            if args.reuse_diffu and temp in diffu_cache:
                diffu_results = diffu_cache[temp]
            else:
                diffu_params = inspect.signature(diffu_generate).parameters
                diffu_extra_kwargs = {}
                if "gen_length" in diffu_params:
                    diffu_extra_kwargs["gen_length"] = args.gen_tokens
                elif "max_new_tokens" in diffu_params:
                    diffu_extra_kwargs["max_new_tokens"] = args.gen_tokens
                diffu_results = diffu_generate(diff_model, diff_tok, temp, questions, diffu_device,
                                               label=f"{diffu_name} diffu", **diffu_extra_kwargs)
                if all(not r["response"] for r in diffu_results):
                    fail(f"all {len(diffu_results)} {diffu_name} responses are empty (generation failing) -- "
                         "aborting instead of saving invalid results")
                if args.reuse_diffu:
                    diffu_cache[temp] = diffu_results
            compare_and_save(judge, ar_results, diffu_results, ar_name, temp, diffu_name, temp, filename)
            done += 1
            print(f"  ({done}/{len(todo)} done, {(time.time() - start) / 60:.1f} min elapsed)", flush=True)
        del ar_model, ar_tok
        free_memory(device)

    print("\n" + "=" * 80)
    print(f"✓ Completed {done} comparisons -> {args.out_dir}/")
    print(f"Next: python consolidate_results.py {args.out_dir}")
    print("=" * 80)


def fail(msg):
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(2)
