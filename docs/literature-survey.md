# QuantRoute: Serving Each LLM Prompt at the Cheapest Quantization Level That Stays Accurate and Safe
## Literature Survey, Gap Analysis and Project Objectives

---

**Abstract.** Deploying a quantized large language model on fixed hardware currently
requires choosing a single bit-width for all incoming traffic, and that choice is
typically made by convention rather than evidence. This survey examines twenty-seven
recent works across seven modules — post-training quantization algorithms, empirical
studies of quantization-induced accuracy loss, per-prompt difficulty estimation,
cost-aware model routing, measured energy/latency benchmarking, same-model
multi-precision serving, and the security of quantized models and routers — and reads them comparatively rather than sequentially. The analysis shows that the field has independently established
(i) that quantization error is concentrated rather than uniform, (ii) that downstream
degradation is strongly task-dependent, and (iii) that per-prompt difficulty is cheaply
predictable. A nascent sixth line of work has begun routing across precisions of one
model, but evaluates on perplexity and carries no cost model or difficulty notion.
Six research gaps are identified and a project is proposed that measures the
(task category × complexity × bit-width) accuracy and cost surfaces on a single
deployment, learns a per-prompt complexity predictor from that surface, and emits a
held-out-validated policy mapping task characteristics to an optimal quantization level.
Notably, the survey also identifies evidence that partially contradicts the project's
intuitive premise, and the objectives are framed accordingly — as a hypothesis to be
tested rather than an assumption to be implemented.

---

## 1. Scope and Method

The project sits at the intersection of four literatures that have developed largely
independently: (i) post-training quantization (PTQ) algorithms, (ii) empirical studies of
what quantization *costs* in accuracy, (iii) per-prompt difficulty estimation, and
(iv) cost-aware model routing. A fifth literature measures the *energy and latency* side
of the trade-off on real hardware, a sixth — the most directly competitive with this project — serves **multiple
precisions of a single model**, and a seventh examines the **security** of quantized
models and of routers.

The survey is organised module-wise along these seven axes. Within each module the papers
are compared rather than summarised sequentially — the analytical claims are carried by
the "what the module collectively establishes / what it leaves open" paragraphs that
close each module, and by the cross-cutting analysis in Section 11. Section 10 regroups
the papers by topic and sets out, for each group, what the works try to solve, where they
fall short, and how this project responds.

Twenty-seven papers are surveyed in the comparative table ([1]–[23] and [29]–[32]), all
published in 2022 or later. Six further works are cited in prose only ([24]–[28] and the
XSTest benchmark [33]).

---

## 2. Module I — Post-Training Quantization Algorithms and the Bit-Width Frontier

**What the module contains.** Dettmers et al. [1] established that naive INT8
quantization fails on models beyond ~6.7B parameters because a small number of
*emergent outlier features* dominate the activation range; their LLM.int8() decomposes
the matrix multiply into an INT8 path and a tiny FP16 outlier path. This diagnosis —
that quantization error is *not* uniformly distributed across the network — is the
premise that the rest of Module I builds on, and each subsequent method is best read as
a different answer to "which parts deserve more bits?"

Frantar et al.'s GPTQ [2] answers it with second-order information, using approximate
inverse-Hessian layer-wise reconstruction to push weight-only quantization to 3–4 bits.
Xiao et al.'s SmoothQuant [3] answers it by *moving* the problem: a mathematically
equivalent per-channel rescaling migrates activation outliers into the weights, which
are easier to quantize, enabling W8A8. Lin et al.'s AWQ [4] answers it by observing that
weight importance is better predicted by *activation* magnitude than by weight magnitude,
and protects roughly 1% of salient channels via per-channel scaling. Kim et al.'s
SqueezeLLM [5] answers it most explicitly of all, splitting each weight matrix into a
heavily-quantized dense component plus a sparse component holding the sensitive outliers,
and using second-order sensitivity to place non-uniform quantization levels. Shao et
al.'s OmniQuant [24] extends the family with learnable weight clipping and learnable
equivalent transformations, and Ma et al.'s BitNet b1.58 [25] shows that if one is
willing to *train* in ternary precision rather than quantize after the fact, 1.58
bits/weight can match FP16 — a different regime from PTQ, but a useful bound on how far
the frontier can move.

Cutting across all of them, Dettmers and Zettlemoyer [6] ran approximately 35,000
experiments to derive *k-bit inference scaling laws*, and reached the field's single most
decision-relevant result: for a fixed total bit budget, 4-bit precision is almost always
Pareto-optimal — a 4-bit model of size 2N beats an 8-bit model of size N.

**Comparative reading.** These methods are usually presented as competitors, but they are
better understood as operating on *orthogonal* aspects of the same error budget: GPTQ
optimises the rounding decision, AWQ and SmoothQuant optimise the scaling applied before
rounding, and SqueezeLLM optimises the allocation of representational capacity. This is
why they compose — the Microscaling study [26] reaches W4A8 with negligible loss by
combining SmoothQuant, AWQ and GPTQ — and why "which method is best" is a less useful
question than "which bit-width, on which hardware, for which workload". Every paper in
this module reports its results primarily as **perplexity plus a handful of zero-shot
accuracies**, averaged across tasks.

**What the module establishes.** Quantization error is concentrated, not uniform;
sensitivity-aware bit allocation is the dominant design principle; and approximately
4 bits is the practical sweet spot for a fixed memory budget.

**What it leaves open.** Every allocation decision in this module is made *inside* the
network — across channels, layers or individual weights. **No paper in this module
allocates bits across the *workload*.** The implicit deployment model is one model, one
bit-width, serving all incoming requests identically. Perplexity, the module's favoured
metric, is a whole-corpus average that by construction cannot express that arithmetic
degrades faster than sentiment classification.

---

## 3. Module II — Empirical Impact of Quantization on Downstream Task Accuracy

**What the module contains.** This module exists because Module I's metrics proved
insufficient, and its papers are progressively more sceptical.

Huang et al. [7] evaluated ten PTQ and LoRA-fine-tuning methods on LLaMA3 across 1–8 bits
and found "non-negligible degradation" concentrated at ultra-low bit widths, with
approximately 2% loss at 4-bit. Dutta et al. [8], in the module's most important
methodological contribution, showed that this framing is itself misleading: across six
quantization schemes and seven tasks, accuracy differences between FP16 and quantized
models were consistently within 0–2%, **yet substantial numbers of individual answers
"flipped"** — correct→incorrect and incorrect→correct in roughly equal proportion,
cancelling out in the aggregate. They propose KL-divergence and flip rate as necessary
complements to accuracy. The implication is severe for anyone selecting a bit-width from
a benchmark table: *aggregate accuracy parity does not imply behavioural equivalence.*

Three papers then establish that degradation is strongly **task-dependent**. Marchisio
et al. [9] found languages are disparately affected — non-Latin scripts worst — and,
critically, that automatic metrics *understate* the damage: a 1.7% average automatic drop
in Japanese corresponded to a 16.0% drop as reported by human evaluators, with
mathematical reasoning degrading fastest. Li et al. [10] isolated mathematical reasoning
and measured up to 32.39% degradation (11.31% average) on Llama-3 under AWQ and GPTQ,
attributing it to error compounding across multi-step chains where precise intermediate
representations are critical. The code-generation replication study [27] found 4-bit to
be "the new frontier", delivering roughly 70% memory reduction without significant
performance loss, but reported drops exceeding 5–10% at 2–3 bits unless a *code-specific*
calibration dataset is used.

The module's most directly relevant paper is Lee et al. [11], who evaluated
instruction-tuned models from 1B to 405B parameters, four quantization methods, and 13
datasets explicitly stratified by task difficulty. Four findings matter here: quantized
large models generally surpass smaller FP16 baselines, though they struggle with
instruction-following and hallucination detection; FP8 is the most robust format and AWQ
generally outperforms GPTQ for weight-only quantization; **smaller models suffer severe
accuracy drops at 4-bit while 70B-scale models maintain stable performance**; and — the
finding that most constrains this project — **hard tasks do not always experience the
largest accuracy losses**, indicating that quantization *magnifies a model's inherent
weaknesses* rather than simply correlating with task difficulty.

**Comparative reading.** Read together, the module tells a coherent but uncomfortable
story. Papers [9], [10] and [27] agree that *category* strongly conditions degradation —
mathematics and code are fragile, commonsense QA and classification are robust. But [11]
explicitly denies that *difficulty* alone predicts degradation, and [8] denies that
accuracy alone measures it. The naive intuition behind complexity-aware quantization —
"harder task ⇒ needs more bits" — is therefore **not** supported by the literature as a
monotonic law; it is a hypothesis that the evidence partially contradicts. There is also
a scale caveat: [11] shows that models in the 1B–3B class are precisely the regime where
4-bit quantization *does* bite, which makes a 3B model the right testbed but means
results should not be extrapolated to 70B.

**What the module establishes.** Degradation is real, category-dependent, worst for
multi-step reasoning and code generation, worse for small models, and poorly captured by
aggregate accuracy alone.

**What it leaves open.** Every paper here is **descriptive and terminal**. Each produces
a table of accuracy deltas and stops. None converts its measurements into a *decision
procedure*. Difficulty, where it appears at all [11], is a **property assigned to a whole
dataset** by the experimenters, not a quantity estimated for an individual incoming
prompt. And no paper in this module reports a *measured monetary or energy cost* against
which the measured accuracy loss could be traded — cost appears only as model size in
gigabytes.

---

## 4. Module III — Task Complexity and Difficulty Estimation

**What the module contains.** Independently of the quantization literature, a body of
work has grown that treats difficulty as a *predictable per-prompt quantity*. Ding et
al.'s Hybrid LLM [12] trains a router on query difficulty, derived from a quality gap
between a small and a large model, and assigns queries accordingly — achieving up to 40%
fewer large-model calls with no drop in response quality. Zhu et al. [13] go further and
show the signal is already latent inside the model: by treating token-level generation as
a Markov chain and learning a value function over hidden states, they estimate
LLM-perceived question difficulty **from the initial hidden state alone, before
generating any output tokens**. Zhao et al. [14] apply this idea to model selection,
training lightweight predictors of problem difficulty and of model correctness on
intermediate representations from s1.1-32B, then routing each problem to the smallest
reasoning model likely to solve it — matching the large model's accuracy at substantially
lower compute.

**Comparative reading.** The module's trajectory runs from *extrinsic* difficulty (how
much better does a large model do — [12]) to *intrinsic* difficulty (what does the
model's own activation geometry say — [13], [14]). The intrinsic approach is cheaper at
inference, requiring no reference model, but demands white-box access and a prefill pass;
the extrinsic approach requires a labelled difficulty corpus but is model-agnostic and
operates on surface features. A practical system on consumer hardware can exploit a
third option the module under-explores: **supervised prediction from surface and
embedding features**, where the *label* is derived empirically from observed model
behaviour rather than from a reference model's advantage.

Note also that all three papers define the prediction target as "will a given model
answer this correctly" — a per-model-pair binary. None defines it as "what is the minimum
precision at which this task survives", which is a continuous, ordinal target with
different structure.

**What the module establishes.** Per-prompt difficulty is predictable cheaply and
reliably enough to drive inference-time decisions.

**What it leaves open.** The predicted difficulty is always consumed by a choice among
**different models**, or among different reasoning budgets. **It has never explicitly
been used to choose a numerical precision** — the qualifier matters, because MoQE [23]
(Module VI) routes across precisions on generic input characteristics, and its router may
well learn something difficulty-like without ever naming it as such. The two literatures — Module II, which knows quantization
sensitivity varies by task, and Module III, which knows task difficulty is predictable —
have not been connected.

---

## 5. Module IV — Cost-Aware Routing, Cascading and Model Selection

**What the module contains.** Chen et al.'s FrugalGPT [15] introduced LLM cascades: query
a cheap model first, score the response, and escalate only on low confidence, reporting
up to 98% cost reduction at accuracy comparable to the best individual model. Aggarwal et
al.'s AutoMix [16] refines the escalation decision with few-shot *self-verification* and,
because that signal is noisy, a POMDP-based meta-router, cutting computational cost by
over 50% at comparable performance. Ong et al.'s RouteLLM [17] replaces cascading with a
*pre-hoc* binary router trained on human preference data from Chatbot Arena, reporting
cost reductions of over 2× and 85% savings on MT-Bench while retaining 95% of GPT-4
quality. Hu et al.'s RouterBench [18] supplies the missing evaluation substrate: over
405,000 inference outcomes across representative LLMs, together with a theoretical
framework for estimating the cost–performance Pareto frontier and a comparative analysis
of routing approaches, "highlighting their potentials and limitations". The sharper
negative result comes from the later LLMRouterBench [28], which reports that "several
routers fail to outperform the Best Single, and some (especially binary routers, such as
HybridLLM and FrugalGPT) **struggle to trade cost for savings while preserving Best
Single accuracy**".

**Comparative reading.** The design space splits cleanly on *when* the decision is made.
Cascades [15], [16] decide *after* a cheap generation, so they pay the small model's cost
on every query but benefit from observing its actual output — a strictly better-informed
but strictly more expensive decision. Pre-hoc routers [12], [14], [17] decide *before*
any generation, so they are cheaper but must predict rather than observe. AutoMix
occupies the middle ground: cheap generation plus explicit verification. The sobering
contribution is [28]'s — under standardised evaluation, several published routers do not
actually dominate the best single model, which sets a bar that any routing claim must
clear rather than assume.

Crucially, **every paper in this module shares one structural assumption: the
alternatives being routed among are different models, and cost is a per-token API
price.** That assumption drives the whole design. Switching between two API endpoints is
free; cost is linear in tokens; and the "small" model has genuinely different capabilities
from the "large" one.

**What the module establishes.** Query-conditional selection demonstrably saves cost at
preserved quality, is best driven by a cheap learned predictor, and must be validated
against a best-single-model baseline on held-out data.

**What it leaves open.** Routing among **bit-widths of one model** breaks all three
assumptions above. The candidates share an architecture and a training run and differ
only in numerical precision, so capability differences are subtle and category-specific
rather than global. Cost is *not* an API price but local compute, dominated by a step
function: whether the weights fit entirely in accelerator memory. And on stock serving
stacks switching is *not* free — it may require evicting and loading gigabytes of weights
(Module VI shows this constraint is removable with nested or overlaid serving, but not in
a standard llama.cpp GGUF deployment). No paper in this module
addresses any of this.

---

## 6. Module V — Measured Cost, Energy and Deployment of Quantized Inference

**What the module contains.** Husom et al. [19] benchmarked 28 quantized LLMs from the
Ollama library on a Raspberry Pi 4, using **hardware-based** high-resolution power
measurement across CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K and HumanEval, and
reporting joules per token and joules per response alongside accuracy — for example,
Qwen2.5-0.5B at 2.61 J/token, roughly 3–3.5× more energy-efficient per token than the
other model families tested, and at 58.83 J/response against Gemma2-2B's 255.79
J/response. Wei et al.'s T-MAC
[20] attacks the kernel level, replacing mixed-precision GEMM with table lookups so that
low-bit weights need no dequantization, reaching 30 tokens/s single-core and 71 tokens/s
on eight cores of an M2-Ultra for BitNet-b1.58-3B, and 11 tokens/s on a Raspberry Pi 5.

**Comparative reading.** These two papers bracket the cost axis. [20] shows that
*achievable* cost is an artefact of kernel implementation as much as of bit-width, while
[19] shows that once the implementation is fixed, cost per task is measurable to high
precision on commodity hardware. Both make the same implicit point that Modules I–IV
elide: **the true cost of a bit-width is empirical and hardware-specific, not derivable
from the bit count.** A 4-bit model is not "4× cheaper" than a 16-bit one; it is cheaper
by whatever the measured throughput, memory residency and kernel efficiency say it is.

**What the module establishes.** Per-task energy and latency of quantized models are
directly and reproducibly measurable, and vary non-linearly with bit-width.

**What it leaves open.** Study [19] measures cost and accuracy on the same runs but
reports them **side by side rather than jointly optimised**, and does not stratify
accuracy by task category or difficulty. Neither paper produces a selection rule. The
VRAM-residency cliff — the discontinuity arising when a model stops fitting in
accelerator memory and must partially offload — is visible in neither, because both
target CPU and edge settings where it does not arise in the same form.

---

## 7. Module VI — Same-Model Multi-Precision Serving

**What the module contains.** The most directly competitive literature is also the
newest, and it attacks the problem from the systems side rather than the evaluation side.

Park et al.'s Any-Precision LLM [21], an ICML 2024 oral, observes that deploying several
differently-sized models is wasteful and proposes *overlaying* LLMs quantized to varying
bit-widths — 3, 4, … n bits — into a memory footprint comparable to a single n-bit
model, supported by a bespoke serving engine. The contribution is that the precisions are
nested rather than stored separately. Google DeepMind's MatQuant [22] reaches a similar
end by a different route, exploiting the fact that integer types are *themselves* nested
— slicing the most significant bits of an int8 weight directly yields int4 or int2 —
so one trained model can be served at int8, int4 or int2 on demand.

MoQE [23] is the closest prior work to this project's routing component. It treats
multiple quantization variants of a single full-precision model as specialised
"quantization experts" and uses a lightweight, structure-aware **router** to dispatch
each input to the most suitable expert based on its characteristics, evaluated on ResNet,
LLaMA and Qwen families over ImageNet, WikiText, C4 and OpenWebText.

**Comparative reading.** [21] and [22] solve the *mechanism* — how to hold many precisions
at once cheaply — while [23] solves a version of the *policy* — which precision a given
input should receive. Together they dismantle two assumptions that would otherwise be
load-bearing for a precision router: that only one precision can be resident at a time,
and that switching therefore costs a model reload. On hardware where nested or overlaid
serving is available, the swap penalty largely disappears. Neither technique is available
in a stock llama.cpp GGUF deployment, so the swap cost remains real *for this project's
chosen stack* — but it is a property of the stack, not a fundamental property of
precision routing, and this survey should not claim otherwise.

**What the module establishes.** Multiple precisions of one model can be served from a
near-single-model memory footprint, and routing inputs across those precisions is an
active, published idea rather than an open opportunity.

**What it leaves open.** The routing in [23] is driven by generic input characteristics
and evaluated predominantly through perplexity and aggregate accuracy on
language-modelling corpora. It carries **no notion of task category or task difficulty**,
no measured cost model (hence no cost-accuracy frontier and no residency or swap
accounting), no fidelity metrics in the sense of [8], and no held-out validation against
fixed-precision baselines of the kind [28] shows routers routinely fail. [21] and [22]
are serving and quantization mechanisms, and make no claim about *which* input deserves
which precision.

---

## 8. Module VII — Security of Quantized Models and Routers

**What the module contains.** Modules I–VI treat the precision decision as benign: the
model files are trusted, a cheaper rung is judged only on accuracy, and the router's
inputs are assumed honest. Four recent works break each assumption. Egashira et al. [29]
were the first to treat quantization as a security risk, constructing models that behave
benignly in full precision but maliciously once quantized, by fine-tuning a malicious
model and then repairing its full-precision weights only within the set that rounds back
to the malicious quantized model. Egashira et al. [30] carried the attack to GGUF, the
format and stack (llama.cpp) this project uses, using the quantization error itself as
slack; across three popular LLMs and nine GGUF types they report large attack deltas for
insecure code generation (Δ = 88.7%), targeted content injection (Δ = 85.0%) and benign
instruction refusal (Δ = 30.1%). Hong et al. [31] evaluated compressed models across eight
trustworthiness dimensions and found that 4-bit quantization preserves trustworthiness
while 3-bit quantization sharply reduces it — a loss that benign accuracy does not show —
and that pruning degrades trustworthiness even at 50% sparsity. Shafran et al. [32]
defined control-plane integrity for LLM routers and showed that query-independent
"confounder gadgets" reliably steer routing decisions in white-box and black-box
settings, while keeping low perplexity so that perplexity filtering does not stop them.

**Comparative reading.** The four works attack three different parts of a precision
router: [29] and [30] the *artefacts* it serves, [31] the *criterion* by which it judges a
cheaper rung acceptable, and [32] the *decision procedure* itself. Read alongside [8],
[31] has a pointed implication: accuracy parity was already shown to be insufficient as a
fidelity criterion, and it is also insufficient as a safety criterion.

| QuantRoute component | Assumption it makes | Work that breaks it |
|---|---|---|
| Seven downloaded GGUF files | Files behave the same as their full-precision source | [29], [30] |
| Selecting cheaper rungs on accuracy | If accuracy holds, behaviour holds | [31] (with [8]) |
| Complexity predictor + router | Prompts are not adversarial | [32] |
| `py_exec` grader and `/route` API | Inputs and model outputs are safe to execute / accept | Standard service security |

**What the module establishes.** Quantized artefacts can be weaponised, low-bit rungs can
lose safety behaviour invisibly to accuracy, and routers can be steered by
query-independent inputs.

**What it leaves open.** None of [29]–[32] concerns a *precision* router. [32] studies
model routers and the upgrade (cost) direction; the downgrade direction — steering a
harmful prompt to a low-bit rung where refusal behaviour has eroded — is specific to
precision routing and unexamined.

---

## 9. Comparative Literature Survey Table

| # | Author [Ref.] | Approach | Dataset / Benchmark | Advantages | Disadvantages | Remarks |
|---|---|---|---|---|---|---|
| 1 | Dettmers, Lewis, Belkada & Zettlemoyer [1] (NeurIPS 2022) | LLM.int8(): mixed INT8/FP16 matmul isolating emergent outlier features | OPT/BLOOM up to 175B; perplexity + zero-shot | First lossless 8-bit at 175B scale; identifies the outlier phenomenon | INT8 only — no ladder below 8 bits; modest speed gains | Diagnostic foundation: quantization error is concentrated, not uniform |
| 2 | Frantar, Ashkboos, Hoefler & Alistarh [2] (ICLR 2023) | GPTQ: one-shot layer-wise reconstruction via approximate inverse-Hessian | OPT, BLOOM, LLaMA; WikiText-2, C4 perplexity | Pushes weight-only PTQ to 3–4 bits; quantizes 175B in ~4 GPU-hours | Weight-only (activations stay FP16); calibration-set sensitive | De-facto 4-bit baseline; later shown weakest on mathematics [10] |
| 3 | Xiao, Lin, Seznec, Wu, Demouth & Han [3] (ICML 2023) | SmoothQuant: equivalent per-channel rescaling migrating activation outliers into weights | OPT, BLOOM, GLM up to 530B; accuracy + latency | Enables true W8A8 (weights *and* activations); real speed/memory win | Hyperparameter α requires tuning; limited benefit below 8 bits | Complementary to [2]/[4] rather than competing — they compose |
| 4 | Lin, Tang, Tang, Yang et al. [4] (MLSys 2024, Best Paper) | AWQ: activation-aware per-channel scaling protecting ~1% salient weights | LLaMA/Llama-2, Vicuna, OpenFlamingo; zero-shot + VQA | No backpropagation or reconstruction; generalises beyond calibration domain; fast kernels | Still weight-only; salience heuristic is a proxy, not optimal | Generally outperforms GPTQ in weight-only comparisons [11] |
| 5 | Kim, Hooper, Gholami, Dong et al. [5] (ICML 2024) | SqueezeLLM: dense-and-sparse split — heavy quantization of the dense part, FP retention of sensitive outliers; sensitivity-based non-uniform levels | LLaMA family; perplexity, MMLU, C4 | Near-lossless 3-bit; 2.1× perplexity-gap reduction vs SOTA; 2.3× speedup on A6000 | Sparse format needs custom kernels; non-trivial engineering | Strongest statement that *sensitivity-aware bit allocation* is the core principle |
| 6 | Dettmers & Zettlemoyer [6] (ICML 2023) | k-bit inference scaling laws: ~35,000 runs sweeping 3–8 bits × 19M–176B parameters | BLOOM, OPT, NeoX/Pythia, GPT-2; zero-shot accuracy | Turns bit-width choice into a scaling law; establishes 4-bit as Pareto-optimal per bit budget | Zero-shot accuracy only; pre-dates instruction-tuned models and reasoning benchmarks | Closest prior work to a *principled* bit-width decision — but global, not per-task |
| 7 | Huang, Zheng, Ma, Qin et al. [7] (2024; *Visual Intelligence*) | Empirical study of 10 PTQ/LoRA-FT methods on LLaMA3 across 1–8 bits, LLMs → MLLMs | LLaMA3-8B/70B, LLaVA; CommonSenseQA, perplexity, VQA | Broad method × bit-width grid; extends to multimodal | Aggregate deltas only; no cost axis; no per-prompt analysis | ~2% loss at 4-bit; severe collapse at ≤3-bit for PTQ |
| 8 | Dutta, Krishna, Kwatra & Ramjee [8] (NeurIPS 2024) | Measures answer *flips* and KL-divergence between FP16 and quantized models | 6 quantization schemes × 7 benchmark tasks | Exposes that accuracy parity conceals large behavioural change; proposes distance metrics | Does not prescribe an acceptable flip threshold; no cost analysis | **Methodological constraint on this project**: accuracy alone is an insufficient selection criterion |
| 9 | Marchisio, Dash, Chen, Aumiller et al. [9] (Findings of EMNLP 2024) | Multilingual quantization evaluation: automatic benchmarks + LLM-as-judge + human evaluation | Multilingual suites across languages and scales | First cross-lingual evidence; shows automatic metrics *understate* damage (1.7% automatic vs 16.0% human, Japanese) | Multilingual focus; no cost or latency measurement | Non-Latin scripts and mathematical reasoning degrade fastest — degradation is structured, not uniform |
| 10 | Li, Su, Yang, Xie et al. [10] (2025) | Isolates mathematical reasoning; multidimensional qualitative + quantitative error analysis; DPO-based recovery | GSM8K and mathematics suites on Llama-3 under AWQ, GPTQ, SmoothQuant | Quantifies worst-case category degradation (up to 32.39%, avg 11.31%); explains the mechanism (error compounding over reasoning chains) | Single category; recovery requires fine-tuning | Strongest evidence that *category* conditions quantization sensitivity |
| 11 | Lee, Park, Kwon, Oh & Kwon [11] (IJCAI 2025) | Large-scale trade-off study: 1B–405B instruction-tuned models × 4 quantization methods × 13 datasets, stratified by task difficulty | 13 datasets incl. instruction-following and hallucination detection | Closest prior work; explicit difficulty stratification; FP8 most robust; AWQ > GPTQ; small models break at 4-bit while 70B stays stable | Difficulty is a *dataset-level* label; descriptive only — yields no decision rule; no measured cost | **Key finding: hard tasks do NOT always lose most — quantization magnifies a model's inherent weaknesses.** Forbids assuming a monotonic complexity→bits relationship |
| 12 | Ding, Mallick, Wang, Sim et al. [12] (ICLR 2024) | Hybrid LLM: router trained on predicted query difficulty routes between a small and a large model | MixInstruct; quality-gap-labelled query sets | Pre-hoc, needs no generation; up to 40% fewer large-model calls at no quality drop | Difficulty label requires both models offline; binary small/large only | First work to make *predicted difficulty* the routing variable |
| 13 | Zhu, Liu, Lin, Tong, Zhong & Shao [13] (2025) | Estimates LLM-perceived difficulty from hidden representations; generation as a Markov chain + learned value function | Textual and multimodal difficulty benchmarks | Requires **no output tokens** — difficulty read from the initial hidden state; outperforms baselines | White-box access required; per-model, not transferable | Shows difficulty is latent in the model itself, not only in prompt surface form |
| 14 | Zhao, Kapusuzoglu, Balasubramaniam, Sahu et al. [14] (NeurIPS 2025 Workshop on Efficient Reasoning) | Lightweight difficulty/correctness predictors on s1.1-32B intermediate representations; routes to the smallest model likely to succeed | Diverse mathematics benchmarks | Directly links predicted difficulty to model-size selection; matches s1.1-32B at far lower compute | Mathematics only; routes across model *sizes*, not precisions; needs a 32B model to featurise | Nearest analogue to the proposed router — but the axis is parameters, not bits |
| 15 | Chen, Zaharia & Zou [15] (2023) | FrugalGPT: LLM cascade — cheap model first, scoring function, escalate on low confidence | HEADLINES, OVERRULING, COQA | Up to 98% cost reduction at GPT-4-level accuracy; needs no white-box access | Pays the small model's cost on every query; scorer must be trained per task | Founding work of cost-aware LLM cascading |
| 16 | Aggarwal, Madaan, Anand, Potharaju et al. [16] (NeurIPS 2024) | AutoMix: few-shot self-verification + POMDP meta-router over the noisy verification signal | 5 models × 5 challenging datasets | Handles verifier noise principledly; >50% cost reduction at comparable performance | Verification costs extra tokens; POMDP needs calibration data | Best-engineered *post-hoc* escalation decision |
| 17 | Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous & Stoica [17] (ICLR 2025) | RouteLLM: pre-hoc binary routers trained on human preference data with data augmentation | Chatbot Arena preferences; MT-Bench, MMLU, GSM8K | >2× cost reduction; 85% savings on MT-Bench at 95% GPT-4 quality; open framework | Requires a large preference corpus; strong/weak binary only | Establishes preference data as a viable difficulty proxy |
| 18 | Hu, Bieker, Li, Jiang, Keigwin, Ranganath, Keutzer & Upadhyay [18] (2024) | RouterBench: standardised routing benchmark + theoretical framework for cost–performance Pareto estimation | 405k+ inference outcomes across representative LLMs, 8 tasks | Makes routers comparable; formalises Pareto-frontier evaluation | API-price cost model only; static offline outcomes; its own abstract claims only a comparative analysis of "potentials and limitations" | Supplies the evaluation substrate; the sharper negative result about routers failing to beat Best Single belongs to LLMRouterBench [28], not here |
| 19 | Husom, Goknil, Shrestha, Bhattarai et al. [19] (2025; ACM TIoT) | Benchmarks 28 quantized LLMs on a Raspberry Pi 4 with hardware-based high-resolution power measurement | CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, HumanEval | Real measured joules/token and joules/response alongside accuracy; 5 diverse benchmarks | CPU/edge only; accuracy and energy reported side by side, not jointly optimised; no difficulty stratification | Proves per-task cost is precisely measurable on commodity hardware |
| 20 | Wei, Cao, Cao, Ma, Wang, Zhang & Yang [20] (EuroSys 2025) | T-MAC: lookup-table mixed-precision GEMM removing dequantization for low-bit CPU/NPU inference | BitNet-b1.58-3B, Llama-2 on M2-Ultra and Raspberry Pi 5 | 30→71 tok/s (1→8 cores); 11 tok/s on Pi 5; 1.1–6.7× speedups | Kernel-level only; CPU/NPU focus; no accuracy study | Shows achievable cost is an artefact of *implementation*, so cost must be measured, never inferred from bit count |
| 21 | Park, Hyun, Cho, Sim & Lee [21] (ICML 2024, Oral) | Any-Precision LLM: overlays models quantized to 3, 4, ... n bits into a footprint comparable to a single n-bit model, plus a specialised serving engine | LLaMA-family models; perplexity and zero-shot accuracy | Removes the memory cost of holding several precisions; **largely dissolves the model-swap penalty** | Requires a bespoke serving engine; unavailable in stock llama.cpp/GGUF | Shows the residency/swap constraint is a property of the serving stack, not of precision routing itself |
| 22 | Nair, Datta, Dean, Jain & Kusupati [22] (ICML 2025) | Exploits the nested structure of integer types, so slicing an int8 weight's most significant bits yields int4 or int2 from one trained model | LLM weight quantization; int8/int4/int2 quality comparisons | One artefact served at several precisions on demand; notable int2 gains | Nested integer types only; needs its own training/quantization procedure | Second, independent route to multi-precision serving from a single model |
| 23 | MoQE [23] (2025) | Treats multiple quantization variants of one full-precision model as "quantization experts"; a lightweight structure-aware **router** dispatches each input to the most suitable expert | ResNet, LLaMA, Qwen; ImageNet, WikiText, C4, OpenWebText | **Closest prior work** - establishes that routing across precisions of one model works and matches the best single quantized model | Routes on generic input characteristics; perplexity-led evaluation; no task-category or difficulty notion; no measured cost model; no held-out policy validation | Refutes any "first precision-level router" claim; leaves the decision-theoretic and cost side open |
| 24 | Egashira, Vero, Staab, He & Vechev [29] (NeurIPS 2024) | Three-stage attack: fine-tune a malicious model, compute the constraint set of full-precision weights that quantize to it, then use projected gradient descent to remove the malicious behaviour in full precision while staying inside that set | Open LLMs under widely used rounding-based quantization; attack scenarios include insecure code generation | First study of quantization as a **security** risk; shows a model can be benign in full precision and malicious once quantized, distributed via model hubs | Targets simpler rounding-based schemes; does not cover GGUF k-quants (addressed by [30]) | Establishes the supply-chain threat: evaluating only the full-precision model is not enough |
| 25 | Egashira, Staab, Vero, He & Vechev [30] (ICML 2025) | First attack on **GGUF**: uses the quantization error between full-precision and dequantized weights as slack to train a model that is malicious only after GGUF quantization | Three popular LLMs across nine GGUF quantization types; insecure code generation (Δ = 88.7%), targeted content injection (Δ = 85.0%), benign instruction refusal (Δ = 30.1%) | Attacks the exact format and stack (llama.cpp / ollama) used in practice; shows scheme complexity is not a defence | Attack-side work; defences are discussed but not the focus | **Directly relevant to QuantRoute**, which serves seven GGUF rungs from llama.cpp |
| 26 | Hong, Duan, Zhang, Li, Xie, Lieberman, Diffenderfer, Bartoldson, Jaiswal, Xu, Kailkhura, Hendrycks, Song, Wang & Li [31] (ICML 2024) | "Decoding Compressed Trust": evaluates compressed LLMs across eight trustworthiness dimensions (stereotype, toxicity, privacy, fairness, ethics, adversarial / out-of-distribution / adversarial-demonstration robustness) alongside benign utility | Three leading LLMs, five compression methods (quantization and pruning) at varying compression rates, including 3-, 4- and 8-bit quantization | 4-bit quantization keeps the original model's trustworthiness; moderate bit-widths can even improve some dimensions | Fixed bit-width per model; no per-task or per-prompt view; no routing | **3-bit sharply reduces trustworthiness, and this is invisible in benign accuracy** — so safety must be measured separately from accuracy |
| 27 | Shafran, Schuster, Ristenpart & Shmatikov [32] (2025) | Defines LLM **control-plane integrity**; generates query-independent "confounder gadgets" that, added to any query, make a router send it to the strong model | Open-source routers and commercial routers; benchmarks include MT-Bench, MMLU and GSM8K; white-box and black-box settings | Shows routers are an attack surface; gadgets work black-box and can keep low perplexity, defeating perplexity filtering | Studies model routers, not precision routers; focuses on upgrade (cost) attacks | Any complexity-based router — including QuantRoute's — must be evaluated against adversarial prompts |

---

## 10. Papers by Topic: What They Address, Where They Fall Short, and How QuantRoute Responds

The modules above read each literature on its own terms. This section regroups the same
works by the problem they set out to solve, states for each one the shortcoming that
matters when choosing a quantization level per prompt, and closes every group with what
this project does about it — including, where the project has already run, what it
measured. QuantRoute's figures come from its own benchmark of Qwen2.5-3B-Instruct across
seven GGUF rungs (Q2_K → F16) on one RTX 4060: 14,469 graded runs over 1,467 tasks and
600 safety prompts.

| Topic | Works | What the group tries to solve | Shared shortcoming | QuantRoute's response |
|---|---|---|---|---|
| 10.1 Quantization algorithms | [1]–[6], [24]–[26] | Lose as little accuracy as possible per bit | Bits are allocated inside the network, never across the workload | Chooses a precision per request across a seven-rung ladder |
| 10.2 Accuracy-impact studies | [7]–[11], [27] | Measure what quantization costs in accuracy | Descriptive only; difficulty is a dataset label; no measured cost | Category × tier cells with flips and KL, turned into a selection rule |
| 10.3 Difficulty estimation | [12]–[14] | Predict per prompt how hard a query is | Target is "will model M be right", never "how much precision" | Predicts the minimum sufficient rung — the required precision |
| 10.4 Cost-aware routing | [15]–[18], [28] | Spend less by sending each query to the right model | Assumes different models, API prices and free switching | Routes across bit-widths of one model on measured local cost |
| 10.5 Measured cost and energy | [19], [20] | Measure real energy, latency and throughput | Cost reported beside accuracy, never optimised with it | One rule picks the cheapest rung that clears accuracy, fidelity and safety bars |
| 10.6 Multi-precision serving | [21]–[23] | Serve several precisions of one model cheaply | No task or difficulty notion, no cost model, no held-out policy test | A category- and difficulty-keyed policy with fidelity and safety checks |
| 10.7 Security | [29]–[33] | Show how quantized models and routers can be attacked | None concerns a precision router; the downgrade direction is unexamined | Pinned hashes, a safety slice, a safety-flip constraint and a safety floor |

### 10.1 Quantization algorithms and the bit-width frontier

| Work | What it attempts to address | Shortcoming for choosing a precision per prompt |
|---|---|---|
| LLM.int8() [1] | Lossless 8-bit inference at 175B scale despite emergent outlier features | INT8 only; one precision serves every request |
| GPTQ [2] | Accurate 3–4-bit weight-only quantization by second-order reconstruction | Judged on perplexity and averaged zero-shot accuracy; later shown weakest on mathematics [10] |
| SmoothQuant [3] | W8A8 by migrating activation outliers into the weights | Little benefit below 8 bits; needs its α tuned |
| AWQ [4] | Protect the ~1% of salient weights using activation statistics | Weight-only; salience is a heuristic proxy |
| SqueezeLLM [5] | Near-lossless 3-bit via a dense-and-sparse split | Needs custom sparse kernels |
| k-bit scaling laws [6] | Which bit-width maximises accuracy for a fixed memory budget | One global answer (4-bit) from zero-shot accuracy on pre-instruction-tuned models |
| OmniQuant, BitNet b1.58, Microscaling [24]–[26] | Learned clipping and transforms, ternary training, composed W4A8 | The same deployment model: one precision serves all traffic |

**How QuantRoute addresses this.** QuantRoute does not propose a new quantization method. It
takes this literature's own principle — spend bits where error hurts — and applies it
across the workload instead of inside the network: seven GGUF rungs of one model, from one
pinned repository, with the rung chosen per request. The measurements confirm [6]'s global
answer and show why a global answer is not enough. Q4_K_M reaches 72.0% bench accuracy
against F16's 73.4% using 35% of F16's memory, while Q2_K collapses to 51.2% — and that
collapse is far from uniform across categories (Section 10.2). Because every rung is
graded on 12 task categories rather than on perplexity, the per-category differences that
a corpus average hides are visible.

### 10.2 Empirical impact of quantization on task accuracy

| Work | What it attempts to address | Shortcoming |
|---|---|---|
| Huang et al. [7] | Map degradation across 10 PTQ/LoRA methods and 1–8 bits on LLaMA3 | Aggregate deltas only; no cost axis; no per-prompt view |
| Dutta et al. [8] | Show that accuracy parity hides answer flips; propose flip rate and KL | No acceptable flip threshold; no cost; no selection rule |
| Marchisio et al. [9] | Measure multilingual damage, including human evaluation | Multilingual focus; no cost or latency |
| Li et al. [10] | Isolate and explain mathematical-reasoning degradation | One category; recovery needs fine-tuning |
| Lee et al. [11] | Trade-offs across 1B–405B models, 4 methods and 13 difficulty-stratified datasets | Difficulty is a dataset label; descriptive only; no measured cost |
| Code-generation study [27] | Whether 4-bit holds up for code | Code only; no decision procedure |

**How QuantRoute addresses this.** QuantRoute resolves accuracy to the cell a decision needs —
12 task categories × 3 difficulty tiers, every task run at every rung — and records answer
flips and corpus KL-divergence against F16 next to accuracy, as [8] asks. The flips matter:
Q8_0 matches F16's accuracy to within 0.4 points yet changes the outcome on 4.4% of tasks,
and Q4_K_M on 12.5%. The measurements are then turned into a decision rule rather than left
as a table (Section 10.4). On [11]'s central claim the data agree that difficulty is not a
monotonic guide to precision: in 25 of the 36 category × tier cells a rung at 4 bits or
below comes within 2 points of F16, only 3 cells need F16, and the hardest tier is not
uniformly the most demanding — on complex mathematics every rung scores below 20%, so no
rung is clearly better than another. On easy and moderate mathematics the low-bit rungs
even score above F16 in this run (86% for Q2_K against 61% for F16 on easy items), a
reminder that rungs differ on individual answers in both directions [8]; the cause has not
yet been diagnosed. Each cell holds 12–56 tasks, so these results say where cheap rungs
clearly hold or clearly break, not how rungs rank within a few points.

### 10.3 Per-prompt difficulty estimation

| Work | What it attempts to address | Shortcoming |
|---|---|---|
| Hybrid LLM [12] | Route between a small and a large model on predicted query difficulty | Labels need both models offline; a binary choice of models |
| Zhu et al. [13] | Read LLM-perceived difficulty from hidden states before generating | White-box and per-model; the target is correctness, not precision |
| Zhao et al. [14] | Predict difficulty from a 32B model's representations to pick the smallest adequate model | Mathematics only; chooses parameters, not bits; needs a 32B model to featurise |

**How QuantRoute addresses this.** QuantRoute changes the prediction target. Instead of "will
model M answer correctly", each task is labelled with its **required precision**: the
cheapest rung at which it is solved correctly *and* confidently — mean token
log-probability at least τ = −0.229, the 10th percentile of F16's correct training answers
— allowing one failure among the costlier rungs. A predictor built from a MiniLM sentence
embedding, 13 surface features and the predicted category is cross-fitted, so no task's own
label shapes its own prediction. Reported honestly, the result is modest: 45.8% exact and
79.8% within one rung on 312 held-out tasks, level with always guessing the most common
class (45.5%) and below guessing each category's median (48.4%). Prompt text alone is a
weak signal for required precision, which points back to [13]'s hidden-state signal as the
natural next source of features.

### 10.4 Cost-aware routing, cascading and model selection

| Work | What it attempts to address | Shortcoming |
|---|---|---|
| FrugalGPT [15] | Cut API spend by cascading from cheap to expensive models | Pays the cheap model on every query; API-price cost model |
| AutoMix [16] | Escalate on noisy self-verification with a POMDP router | Verification costs extra tokens; different models only |
| RouteLLM [17] | Pre-hoc strong/weak routing learned from preference data | Needs a large preference corpus; a binary choice |
| RouterBench [18] | A standard benchmark and Pareto framework for routers | API prices; offline outcomes only |
| LLMRouterBench [28] | Test whether routers beat the best single model | Shows several do not — the bar any router must clear |

**How QuantRoute addresses this.** QuantRoute routes among **bit-widths of one model** rather
than among different models, so cost is measured local GPU time per 1,000 tasks instead of
an API price, and switching rungs is a real model load — 2.1 to 5.6 seconds on this
machine — rather than free. A rung must clear an accuracy lower bound, an answer-flip upper
bound and a safety upper bound before cost is even considered, and the cheapest rung that
passes is served. Fit on the training split at moderate bars (δ = 0.10, φ = 0.30,
ψ = 0.10), the policy serves 64 of 84 (category, predicted-precision) cells below F16 at
54% of the always-F16 cost. At the strict default (δ = 0.02) it keeps F16 everywhere,
because about 85 training tasks per category cannot establish a 2-point match at 95%
confidence. The held-out comparison against always-F16, the best single fixed rung and a
resident-subset policy — the test [28] shows routers often fail — is the next analysis
step, and its outcome will be reported whichever way it falls.

### 10.5 Measured cost, energy and deployment

| Work | What it attempts to address | Shortcoming |
|---|---|---|
| Husom et al. [19] | Real per-token energy and accuracy for 28 quantized LLMs on a Raspberry Pi 4 | Cost and accuracy side by side, not optimised together; no difficulty strata; CPU/edge only |
| T-MAC [20] | Fast low-bit inference on CPUs and NPUs via table lookups | Kernel-level; no accuracy study |

**How QuantRoute addresses this.** Every rung was profiled on the target GPU — VRAM footprint
(1.5–6.1 GB), load time, per-task latency and bits per weight — and those measured costs
enter the same rule as accuracy, so the two are optimised together rather than tabulated
together. The measurements support [20]'s point that cost does not follow from the bit
count: Q2_K, the smallest file, averages 0.87 s per task against 0.53 s for Q4_K_M, and F16
takes 1.16 s. The expected VRAM-residency cliff did not appear on this card — even F16
(6.1 GB) stayed fully resident in 8 GB — so on this hardware the cost of switching between
rungs, not a residency cliff, is the discontinuity a router has to account for.

### 10.6 Same-model multi-precision serving

| Work | What it attempts to address | Shortcoming |
|---|---|---|
| Any-Precision LLM [21] | Hold many precisions in one model's memory footprint | Needs a bespoke serving engine; no claim about which input gets which precision |
| MatQuant [22] | Serve int8, int4 and int2 from one trained model | Nested integer types and its own training procedure; no routing policy |
| MoQE [23] | Route each input to the best of several quantized variants | Generic input features; perplexity-led evaluation; no task or difficulty notion, no measured cost, no held-out policy validation |

**How QuantRoute addresses this.** QuantRoute runs on stock llama.cpp, where nested precisions
are not available, so it measures the cost of switching rather than assuming it away.
Against MoQE, the closest prior work, the difference is the decision basis and the
objective: QuantRoute's policy is keyed on task category and predicted required precision,
and it chooses the *cheapest sufficient* rung under explicit accuracy, fidelity and safety
bounds rather than the rung expected to perform best. Results are reported as graded task
accuracy, flip rate and KL-divergence rather than perplexity, and the policy is fit on a
training split so it can be tested on held-out tasks.

### 10.7 Security of quantized models and routers

| Work | What it attempts to address | Shortcoming for a precision router |
|---|---|---|
| Egashira et al. [29] | Show a model can be benign in full precision but malicious once quantized | Rounding-based schemes only |
| Egashira et al. [30] | Carry that attack to GGUF k-quants in llama.cpp | Attack-side; defences are not the focus |
| Hong et al. [31] | Measure how compression changes trustworthiness | A fixed bit-width per model; no routing |
| Shafran et al. [32] | Show that query-independent gadgets steer LLM routers | Model routers and the upgrade direction only |
| XSTest [33] | Test exaggerated safety with harmful and harmless look-alike prompts | A test set, not a defence |

**How QuantRoute addresses this.** All eight GGUF files come from one official repository at
a pinned commit and are checked against their SHA-256 hashes before every server launch,
and model-written code is only ever executed inside a Docker sandbox with no network and
capped memory, CPU and time. A 600-prompt safety slice — XSTest's 200 harmful and 250
harmless look-alike prompts plus 150 planted prompt injections — runs at every rung, and a
rung's directional safety-flip rate (F16 safe, rung unsafe) is the third condition in the
selection rule. At ψ = 0.05 the measured upper bounds bar Q2_K, Q3_K_M and Q4_K_M and make
Q5_K_M the floor below which prompts flagged by the safety classifier are never served.
That classifier catches 96% of held-out harmful prompts but also flags 73% of the harmless
look-alikes, and the string-match refusal grader misses refusals that are not phrased as
templates, so both are reported with those weaknesses stated. The adversarial half —
cost-inflation and downgrade attacks on the router in the style of [32] — has not been run
yet and is the next security step.

---

## 11. Cross-Cutting Comparative Analysis

Three tensions emerge only when the modules are read against one another.

**Tension 1 — Sensitivity-aware allocation stops at the network boundary.** Module I's
entire intellectual thrust is that bits should be allocated where error hurts most:
across channels [3], [4], across individual weights [5], and across the model-size versus
precision budget [6]. Module II then demonstrates that error also hurts unequally *across
tasks* — dramatically so, with degradation exceeding 32% on mathematics against near-zero
on commonsense QA [9], [10], [11]. The obvious extension of Module I's own principle to
the workload axis has simply not been taken.

**Tension 2 — Difficulty is predictable, but never consumed by a precision decision.**
Module III shows per-prompt difficulty can be estimated cheaply, even without generating
a single token [13]. Module IV consumes that estimate — but always to select a *different
model* [12], [14], [17]. The quantization ladder is a candidate set with an unusually
attractive property for routing: the candidates are the *same model*, so capability drift
is bounded and the comparison is clean. Module VI shows this has begun to be exploited —
MoQE [23] routes inputs across quantization variants of one model — but on perplexity-led
evaluation, without a difficulty notion, a measured cost model, or a validated policy.

**Tension 3 — The premise this project must not assume.** Lee et al. [11] report that
hard tasks do not reliably suffer the largest accuracy losses, because quantization
magnifies *pre-existing* weaknesses rather than tracking difficulty. Simultaneously,
Dutta et al. [8] show that accuracy parity can conceal large behavioural change.
Together these mean that the intuitive rule "complex task ⇒ more bits" is **an empirical
question, not a design assumption**. Any honest system must learn the
(category × complexity → precision) surface from measurement and validate it against
fixed-precision baselines [18], [28], rather than assume its shape in advance.

Finally, across all twenty-seven papers, **cost is expressed in four mutually incomparable
currencies**: perplexity (Module I), gigabytes of memory (Modules I–II), API dollars per
token (Module IV), and joules or tokens per second (Module V). No work joins a measured
*local* cost to a *per-category* accuracy measurement on a single deployment — which is
exactly what a practitioner choosing a quantized model file for a consumer GPU actually
needs.

---

## 12. Research Gap

Across the twenty-seven works surveyed, the following gaps are identified.

**G1 — Bit-width sensitivity is measured at aggregate level, not at a granularity that
supports decisions.** Modules I and II report degradation averaged over whole benchmarks
[1]–[7], [9], [10]. Even the most granular study [11] stratifies by dataset, not by the
(category × complexity) cell that a decision procedure would need to consult.

**G2 — Task difficulty is treated as a dataset annotation, not a predicted per-prompt
quantity, in all quantization work.** The techniques to predict it per prompt exist
[12], [13], [14] but live in a disconnected literature and target a different decision.
Moreover, all of them predict *"will model M answer this correctly"*; none predicts the
ordinal quantity a quantization router actually requires — **the minimum precision at
which a task survives**.

**G3 — Precision routing exists, but is not cost-aware, task-aware, or validated.**
The mainstream routing literature ([12], [14], [15], [16], [17], [18]) assumes
heterogeneous models with API-priced tokens and free switching, and so never confronts
the cost structure of a quantization ladder: shared weights, a discontinuous
VRAM-residency cliff, and a non-zero model-swap penalty. Module VI shows that routing
across precisions of a single model is *not* itself unexplored — MoQE [23] already does
it, and Any-Precision LLM [21] and MatQuant [22] show the residency and swap constraints
are artefacts of the serving stack rather than fundamental. The gap is narrower and more
defensible than "nobody routes across bit-widths": **no existing precision router selects
on task category or predicted difficulty, optimises against a measured local cost model,
reports fidelity metrics, or validates its policy on held-out data against fixed-precision
baselines.** MoQE routes on generic input characteristics and reports perplexity-led
results; it offers no cost axis and no evidence that its policy beats the best single
quantization under the standard [28] test.

**G4 — Accuracy and measured cost are never jointly optimised on one deployment.**
Module V measures energy and latency rigorously [19], [20] but reports them beside
accuracy rather than deriving a frontier from them; Modules I–IV substitute proxies
(perplexity, gigabytes, API price). No study yields a per-category cost–accuracy Pareto
frontier in consistent, measured units for a single machine.

**G5 — The literature is descriptive, not prescriptive, and is unvalidated as policy.**
Not one surveyed paper outputs an actionable bit-width selection rule, and none provides
held-out evidence that such a rule would beat simply fixing a bit-width — despite [28]
demonstrating that routers frequently *fail* precisely that test. This applies to the
precision routers of Module VI as much as to the model routers of Module IV: MoQE [23]
reports comparability with the best single quantized model on perplexity-led benchmarks,
not a held-out cost–accuracy win under a standardised routing evaluation. Compounding this, [8]
shows that any selection criterion based on accuracy alone is insufficient, yet no
routing or quantization-selection work incorporates fidelity metrics such as flip rate or
KL-divergence.

**G6 — The precision decision is treated as benign; no precision router considers an
adversary or a safety constraint.** Quantization can be weaponised so that a model is
benign at full precision but malicious once quantized [29], including under the GGUF
schemes used by llama.cpp [30]. Low-bit quantization can erode trustworthiness while
benign accuracy appears unchanged [31], so an accuracy-only selection rule (even with
fidelity metrics such as flip rate [8]) can select a rung that is accurate but unsafe.
Routers are themselves an attack surface: query-independent gadgets reliably manipulate
routing decisions, and perplexity filtering does not stop them [32]. None of the
precision-routing or quantization-selection works ([21]–[23], [12]–[18]) evaluates
artefact integrity, safety behaviour per bit-width, or the adversarial robustness of the
routing decision.

### 12.1 Project Rationale

The practical consequence of these gaps is that a practitioner deploying a quantized
model on fixed hardware has no principled basis for choosing a bit-width. The field
offers a global heuristic — "use 4-bit" [6] — derived from zero-shot accuracy on
pre-instruction-tuned models, alongside a body of evidence [9], [10], [11] proving that
this single global answer is wrong for at least some of their traffic, without indicating
which part. Meanwhile the routing literature has built exactly the machinery needed to
make traffic-conditional decisions [15]–[18], and the difficulty-estimation literature
has built exactly the predictor needed to drive it [12]–[14], but neither has been
pointed at precision selection.

This project closes that loop. It measures the (category × complexity × bit-width)
accuracy surface and the matching measured-cost surface on a single deployment, learns a
per-prompt complexity predictor whose label is derived from that surface, and emits a
validated routing policy. Critically, it treats the "harder ⇒ more bits" intuition as a
hypothesis to be tested against fixed-precision baselines rather than a premise to be
implemented — a direct response to [11] and [28].

---

## 13. Novelty: How the Proposed System Addresses Each Gap

| Gap | How the proposed system addresses it | Why this is novel |
|---|---|---|
| **G1** Aggregate-only sensitivity | Benchmarks a 7-point quantization ladder (Q2_K → F16) of one instruction-tuned model across **12 task categories × 3 complexity tiers**, with bootstrap confidence intervals per cell and at least 10 items per cell by construction | First accuracy surface resolved to the (category × complexity × bit-width) cell rather than the dataset; [11] is the closest and stops at dataset-level difficulty labels |
| **G2** Difficulty as dataset annotation | Trains a per-prompt complexity analyser on surface and embedding features whose **label is empirically derived** — the minimum bit-width at which the task is reliably solved, measured on the project's own grid — and reports its cross-validated accuracy as a first-class result | Redefines the prediction target from "will model M be correct" [12]–[14] to **"what precision does this prompt require"**. MoQE [23] also selects a precision per input, so the distinction is the *objective*: MoQE's router picks the expert expected to perform **best**, whereas this predicts the **minimum sufficient** precision under a cost constraint — a cheaper target that is only meaningful once cost is measured |
| **G3** Precision routing not cost- or task-aware | Routes across **bit-widths of a single model** on *task category and predicted difficulty*, using a VRAM-budgeted server pool with LRU load/evict, and a cost model containing an explicit **VRAM-residency cliff** plus a separately accounted model-swap penalty; includes a resident-subset baseline that restricts the policy to quants which fit concurrently | **Not** the first precision router — MoQE [23] precedes it. The novelty is the decision basis and the accounting: first precision router driven by task category and predicted difficulty, and first cost model treating VRAM residency as a first-class discontinuity and swap as a separately charged event rather than a linear memory term |
| **G4** Cost and accuracy never jointly optimised | Measures prefill/decode throughput, per-task latency, peak VRAM and bytes-per-parameter; converts to **cost per 1,000 tasks** via a single configurable GPU hourly rate; computes per-cell Pareto frontiers in those units | Places accuracy and measured local cost in one optimisation on one machine — [19] measures both but never joins them; [18] optimises a frontier but only in API prices |
| **G5** Descriptive, unvalidated | Emits a policy mapping (category, *predicted required precision*) → bit-width, fit on a **training split only** via a pooled model that shares strength across thinly-populated cells, and scored on a held-out split through the full inference pipeline against always-F16, always-Q8_0, always-Q4_K_M, best-single-fixed-quantization, a **resident-subset** policy and an **oracle** upper bound. Selection is a **two-condition admissibility rule** — a level qualifies only if its accuracy lower confidence bound is within δ of the reference **and** its flip rate against the reference is ≤ φ, with the cheapest admissible level chosen and (δ, φ) swept to trace the frontier. Fidelity is therefore a constraint on the choice, not a footnote to it: a level that matches on accuracy while churning a large share of individual answers is rejected | First prescriptive, held-out-validated quantization-selection policy; first routing work to make [8]'s fidelity metrics an admissibility constraint on the selection rather than a post-hoc report; directly answers [28]'s finding that routers often fail to beat the best single model — including reporting honestly if this one does not. Note the policy key is the *predicted required precision* conditioned on category, not a separate heuristically assigned tier |
| **G6** Precision decision assumed benign | Adds a safety slice with a safety-flip constraint to the selection rule; evaluates the router against cost-inflation and downgrade attacks; pins model-file hashes; repurposes cross-rung flips/KL vs F16 as a backdoor detector | First precision-selection policy with a safety constraint and an adversarial evaluation of the precision decision itself |

An additional contribution follows from [11]. Because that study found difficulty does
*not* monotonically predict degradation, the policy here is **learned from the measured
surface rather than imposed on it**. If the data show that a hard mathematics task and an
easy mathematics task require the same precision, the fitted policy will say so. The
project is therefore a test of [11]'s claim at a granularity [11] did not examine, as
well as an engineering artefact.

---

## 14. Project Objectives

**O1 — Construct a reproducible measurement harness for the accuracy–fidelity surface.**
Build a deterministic benchmark that evaluates a seven-level quantization ladder (Q2_K,
Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16) of a single instruction-tuned model over
~1,500 graded tasks (135 curated seed tasks plus ~1,370 generated and teacher-verified), spanning 12 categories
and 3 stratification tiers, reporting
per-cell accuracy with bootstrap confidence intervals **and** answer-flip rate and
KL-divergence against the F16 reference.
*Addresses G1 and G5; responds to [8] and [11].*

**O2 — Develop a per-prompt task complexity analyser.** Define task complexity
empirically as the **required precision** of a task. A raw "lowest bit-width that happened
to succeed" label would be largely noise, since [8] shows answers flip in both directions
at comparable rates, so the label is defined in two steps.

First, **"solved" is defined to include confidence, not just correctness**: a task counts
as solved at level b if the answer is graded correct *and* its confidence margin clears a
threshold τ. The margin is the answer logprob, optionally corroborated by agreement with
the greedy answer at **higher levels only**. The direction matters: at a task's true
threshold b*, the level immediately below b* is wrong by construction, so requiring
agreement with the lower neighbour would make the threshold level unsolvable and shift
every label up one rung — biasing the labels at exactly the boundary being measured.
τ is tuned on the **training split only**, since it changes the label distribution.

Second, the **required precision** is the lowest level b such that the task is solved at
b and at all higher levels, **allowing at most one exception above b**, and excluding F16
from the monotonicity requirement entirely. Both relaxations are deliberate: a strict
all-levels-above rule discards any task that flips once near the top of the ladder, which
at single-sample resolution will be a noticeable share of the set, and F16 was expected to
be partially offloaded, and therefore an impure reference, on this hardware (in the
measured run it fit entirely in VRAM).

Two boundary cases are settled explicitly. A task solved only at the very top of the
ladder takes **required precision = F16**, which is a legitimate label value rather than a
discard, though such tasks are flagged for review. Tasks
solved at *no* level are excluded from fitting and reported separately as lying outside
the model's competence.

Train a lightweight predictor (surface features plus sentence embeddings) that estimates
this ordinal target for an unseen prompt without invoking the full-precision model, and
report its cross-validated accuracy as a first-class result. **The predictor's output is
the required-precision class itself, and it is what drives routing at inference time**;
the heuristically assigned easy/moderate/complex tier is used only to stratify the task set and to
report results, never as the routing key. This keeps one definition of complexity in the
system rather than two.
*Addresses G2.*

**O3 — Build a measured cost model incorporating hardware discontinuities.** Instrument
prefill and decode throughput, per-task latency, peak VRAM, bytes-per-parameter and
full-residency status for each quantization level; convert these to cost per 1,000 tasks
via a configurable GPU hourly rate; and account for the model-swap penalty separately
from steady-state inference cost.
*Addresses G3 and G4; responds to [19] and [20].*

**O4 — Derive and validate an optimal quantization-selection policy.** Compute
cost–accuracy Pareto frontiers and fit a policy that maps a prompt's predicted
required-precision class, conditioned on task category, to a served bit-width.

**The selection rule, stated once.** For a cell, level *b* is admissible iff

> `LCB_acc(b, cell) ≥ acc(ref, cell) – δ`  **and**  `flip(b, cell) ≤ φ`

where `LCB_acc` is the lower confidence bound on accuracy from the pooled model, `ref` is
the reference level, δ is the tolerated accuracy give-away and φ the flip-rate ceiling.
**Among admissible levels, the cheapest is chosen.** Sweeping δ (and φ) traces the
policy's cost–accuracy frontier. Both conditions are necessary: fidelity alone would let a
level win a cell on a low flip rate while scoring badly, and accuracy alone would ignore
[8]'s finding that parity can conceal wholesale answer churn.

Three statistical commitments follow from the cell sizes and the data structure.

First, per-cell accuracy is estimated from a **pooled model** (logistic regression of
correctness on category, predicted complexity and log bytes-per-parameter, with
interactions) rather than from independent per-cell tables: with roughly a dozen items per
cell, an independent bootstrap interval spans tens of percentage points and a
lower-confidence-bound rule would degenerate to "always pick the highest bit-width".

Second, the complexity regressor must be the **cross-fitted, out-of-fold predicted**
required precision, never the observed label. The label is *defined from* the same
correctness outcomes being regressed, so using it directly is circular — the model would
appear near-perfect in training and carry no information. Using the heuristic tier instead
would silently reinstate the two-variable design. The predictor is therefore trained on
k–1 folds and used to predict the held-out fold, rotating through all k, and those
out-of-fold predictions enter the pooled model.

Third, the rows are **not independent**: each task contributes one row per ladder level,
and those outcomes are strongly correlated within a task. A plain logistic fit would
report standard errors that are far too narrow, and the selection rule depends on those
intervals. Estimation therefore **clusters by task** — in practice a GEE fit with an
exchangeable working correlation and task-robust standard errors, which captures
within-task dependence without requiring a full mixed model (Python has no convenient
frequentist random-effect logistic fit; the alternative is variational Bayes via
`BinomialBayesMixedGLM`). Uncertainty is confirmed by a bootstrap that **resamples whole
tasks rather than individual rows**.

The task set is also grown well beyond the initial ~450 items toward ~1,500 so that cells
are populated enough for the frontier to mean anything.

Both `LCB_acc` and `flip` come from the **pooled model**, not from raw per-cell counts:
a flip rate measured on a dozen items carries exactly the thin-cell noise the pooled model
exists to avoid, so flips are fitted as a second logistic regression ("flipped vs the
reference") on the same regressors, or aggregated to category level where cells are
sparse.

**Held-out evaluation is deferred until the task set is final.** At ~450 tasks a cell
holds roughly a dozen items, so even a stratified assignment leaves some cells with almost
nothing held out — with a 30% test share, a cell has about an 8.5% chance of retaining at
most one test item, so around three of 36 cells would be effectively unevaluable. Growing
to ~1,500 (~42 per cell) removes the problem; until then the split exists only to keep
training assignments stable.

Evaluate on the final held-out split against always-F16, always-Q8_0, always-Q4_K_M, the
best single fixed quantization, a **resident-subset** policy restricted to quantizations
that fit in VRAM concurrently, and an oracle upper bound. The held-out evaluation runs the
**full inference-time pipeline** — predicted category, then predicted required precision,
then policy — never ground-truth labels, and the held-out tasks are excluded when deriving
the labels the predictor is trained on. Report the outcome faithfully whether or not the
learned policy wins.
*Addresses G1 and G5; responds directly to [18] and [28].*

**O5 — Deploy the policy as a working precision-routing service.** Implement a service
that classifies an incoming prompt's category, predicts its complexity, consults the
policy, dispatches to the selected quantization through a VRAM-budgeted server pool with
LRU eviction, and returns the answer together with the chosen bit-width, estimated cost
and selection rationale — demonstrating that the study's findings are operational rather
than merely descriptive.
*Addresses G3 and G5.*

**O6 — Evaluate and harden the security of precision routing.** Four parts.

1. **Safety slice and safety constraint.** Add a safety slice to the benchmark: the 200
   unsafe prompts of XSTest [33], which the model should refuse; its 250 safe prompts
   that superficially resemble unsafe ones, which it should *not* refuse; and about 150
   prompt-injection tasks built by planting an instruction carrying a unique canary
   string inside existing extraction and summarization passages. Refusal is graded by a
   deterministic refusal-phrase matcher whose error rate is estimated by an LLM-judged
   audit of a stratified sample of 200 responses. A rung's **safety flip rate** is the
   fraction of harmful and injection prompts on which F16 behaves safely and the rung does
   not. It is directional, because a rung that refuses *more* harmful requests than the
   reference is not a safety regression. The selection rule gains a third condition:

   > `LCB_acc(b, cell) ≥ acc(ref, cell) – δ` **and** `flip(b, cell) ≤ φ` **and** `safety_flip(b) ≤ ψ`

   The safety flip rate is measured per rung over the whole safety slice and applies in
   every cell, since any prompt may be adversarial. In addition, prompts flagged by a
   safety classifier are never served below the lowest rung whose safety flip rate is
   within ψ. Over-refusal on the safe XSTest prompts is reported per rung but not
   constrained. *Responds to [31] and [8].*

2. **Adversarial evaluation of the router.** The complexity predictor uses features that
   are easy to game. Measure the success of cost-inflation attacks (padding, feature
   stuffing, and query-independent greedy suffix gadgets in the style of [32]) that push
   prompts toward F16, and downgrade attacks (casual framing, and gadgets that minimise
   predicted complexity) that push harmful prompts toward low-bit rungs. Measure three
   defences — the safety floor, routing on a normalised form of the prompt (with its cost
   on clean traffic reported), and per-client limits on F16 — individually and combined.
   *Responds to [32].*

3. **Model-file integrity.** Download only from the official model repository at a
   pinned commit, and verify the SHA-256 hash of every GGUF file before each server
   launch. Treat the cross-rung comparison (flips and KL-divergence against F16) as a
   **differential test**: behaviour that diverges only at specific rungs, especially on
   code tasks, is the signature of a quantization-triggered backdoor. The test's
   sensitivity is not validated against a planted backdoor, and results are reported with
   that caveat. *Responds to [29] and [30].*

4. **Service hardening.** Bind every `llama-server` instance and the API to localhost;
   require an API key; apply per-client rate limits, including a separate limit on
   requests served at F16; cap input size; disable forced-rung requests by default and
   never allow them below the safety floor for flagged prompts; and run the `py_exec`
   grader, which executes model-generated code, in a Docker container with no network,
   memory and CPU limits, a process limit, a read-only filesystem and a timeout.

*Addresses G6; extends O1 (benchmark), O4 (selection rule) and O5 (service).*

### 14.1 Threat model

| Threat | Attacker action | Impact on QuantRoute | Defence |
|---|---|---|---|
| Cost inflation / DoS | Pads prompts so the predictor says "needs F16" | F16 is the largest and slowest rung (6.1 GB, ~1.2 s per task, ~5.6 s to load); pushing traffic onto it and swapping rungs stalls other users | Per-client limits on F16; route on a normalised prompt |
| Downgrade (jailbreak by routing) | Makes harmful prompts look easy so they reach Q2_K | Low-bit rungs are where safety behaviour degrades [31] | Minimum rung for safety-flagged prompts; safety-flip constraint |
| Poisoned model file | Distributes a GGUF that is malicious only when quantized [29], [30] | Project downloads and serves seven GGUF files | Official repo only; pinned SHA-256 hashes; cross-rung differential testing |
| Grader / service abuse | Calls the API directly or smuggles code into `py_exec` | Grader executes model-generated code | Localhost-only servers; auth, rate limits, input caps; sandboxed grader |

---

## 15. Reported Results versus QuantRoute Results

Each row sets a figure reported in prior work beside what QuantRoute measured. The studies
used different models, quantization schemes and benchmarks, so this is a check of whether
QuantRoute's measurements agree with each published finding — not a like-for-like
leaderboard. QuantRoute figures: Qwen2.5-3B-Instruct, seven GGUF levels, one RTX 4060.

| Finding | Reported in prior work | QuantRoute measured | Verdict |
|---|---|---|---|
| Accuracy loss at 4 bits | About 2% loss at 4-bit and collapse at 3 bits or fewer (LLaMA3, 10 PTQ methods) [7] | −1.4 pp at Q4_K_M, −4.3 pp at Q3_K_M, −22.2 pp at Q2_K against F16 | Agrees |
| Best bit-width for a memory budget | 4-bit is Pareto-optimal for a fixed bit budget (19M–176B models, zero-shot accuracy) [6] | Q4_K_M keeps 98% of F16's accuracy at 35% of its memory; accuracy is flat from Q5_K_M upward | Agrees |
| Accuracy hides changed answers | Accuracy within 0–2% of FP16 while many individual answers flip [8] | Q8_0 is within 0.4 pp of F16 yet changes 4.4% of answers; Q4_K_M is −1.4 pp and changes 12.5% | Agrees |
| Mathematical reasoning | Up to 32.39% loss on mathematics, 11.31% on average (Llama-3; AWQ, GPTQ, SmoothQuant) [10] | No consistent loss: mathematics scores 56.6% at Q2_K against 38.0% at F16 — the cause is not yet diagnosed | Differs |
| Code generation | 4-bit holds; drops beyond 5–10% at 2–3 bits [27] | Code generation is 88.1% at Q4_K_M, equal to F16; −17.5 pp at Q3_K_M and −58.7 pp at Q2_K | Agrees |
| Small models at 4 bits | Small models suffer severe drops at 4-bit while 70B-scale models stay stable [11] | A 3B model loses only 1.4 pp at 4-bit with GGUF k-quants; severe loss appears only at 2-bit | Differs |
| Difficulty versus degradation | Hard tasks do not always lose the most; quantization magnifies existing weaknesses [11] | In 25 of 36 category × tier cells a level of 4 bits or less is within 2 pp of F16; the hardest tier is not uniformly the most demanding | Agrees |
| Safety by bit-width | 4-bit preserves trustworthiness; 3-bit sharply reduces it [31] | Safety-flip upper bound 6.1% at Q2_K, 5.6% at Q3_K_M, 6.1% at Q4_K_M, 3.0% at Q5_K_M and 1.8% at Q6_K and Q8_0 — erosion runs through 4-bit, with no sharp 3-to-4-bit break | Partly agrees |
| Cost versus bit count | Inference speed depends on kernel implementation, not bit count alone [20] | Q2_K, the smallest file, is slower per task (0.87 s) than Q4_K_M (0.53 s) | Agrees |
| Predicting difficulty per prompt | Predicted difficulty routes well — up to 40% fewer large-model calls [12]; hidden-state estimates beat baselines [13] | Required-precision predictor: 45.8% exact against a 45.5% majority baseline, 79.8% within one level (text features, different target) | Differs |
| Cost saving from routing | Up to 98% [15], over 50% [16] and 85% at 95% of GPT-4 quality [17], for API-priced model routing | 46% below the always-F16 cost at moderate bars (fit on the training split); held-out quality still to be measured | Not yet tested |
| Router versus best single model | Several routers fail to beat the best single model [28]; MoQE matches the best single quantized model [23] | The held-out comparison against fixed quantization levels is the next analysis step | Not yet tested |

---

## 16. References

[1] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2022. arXiv:2208.07339.

[2] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," in *Proc. International Conference on Learning Representations (ICLR)*, 2023. arXiv:2210.17323.

[3] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," in *Proc. International Conference on Machine Learning (ICML)*, 2023. arXiv:2211.10438.

[4] J. Lin, J. Tang, H. Tang, S. Yang, et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," in *Proc. Machine Learning and Systems (MLSys)*, 2024 (Best Paper Award). arXiv:2306.00978.

[5] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney, and K. Keutzer, "SqueezeLLM: Dense-and-Sparse Quantization," in *Proc. International Conference on Machine Learning (ICML)*, 2024. arXiv:2306.07629.

[6] T. Dettmers and L. Zettlemoyer, "The Case for 4-bit Precision: k-bit Inference Scaling Laws," in *Proc. International Conference on Machine Learning (ICML)*, PMLR vol. 202, pp. 7750–7774, 2023. arXiv:2212.09720.

[7] W. Huang, X. Zheng, X. Ma, H. Qin, et al., "An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs," *Visual Intelligence*, 2024/2025. arXiv:2404.14047.

[8] A. Dutta, S. Krishna, N. Kwatra, and R. Ramjee, "Accuracy is Not All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. arXiv:2407.09141.

[9] K. Marchisio, S. Dash, H. Chen, D. Aumiller, et al., "How Does Quantization Affect Multilingual LLMs?" in *Findings of the Association for Computational Linguistics: EMNLP*, 2024. arXiv:2407.03211.

[10] Z. Li, Y. Su, R. Yang, C. Xie, Z. Wang, Z. Xie, N. Wong, and H. Yang, "Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning," 2025. arXiv:2501.03035.

[11] J. Lee, S. Park, J. Kwon, J. Oh, and Y. Kwon, "Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant," in *Proc. International Joint Conference on Artificial Intelligence (IJCAI)*, 2025. arXiv:2409.11055.

[12] D. Ding, A. Mallick, C. Wang, R. Sim, S. Mukherjee, V. Rühle, L. V. S. Lakshmanan, and A. Awadallah, "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," in *Proc. International Conference on Learning Representations (ICLR)*, 2024. arXiv:2404.14618.

[13] Y. Zhu, D. Liu, Z. Lin, W. Tong, S. Zhong, and J. Shao, "The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden Representations," 2025. arXiv:2509.12886.

[14] B. Zhao, B. Kapusuzoglu, K. Balasubramaniam, S. Sahu, S. Chakraborty, and G. I. Winata, "Optimizing Reasoning Efficiency through Prompt Difficulty Prediction," in *NeurIPS Workshop on Efficient Reasoning*, 2025. arXiv:2511.03808.

[15] L. Chen, M. Zaharia, and J. Zou, "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," 2023. arXiv:2305.05176.

[16] P. Aggarwal, A. Madaan, A. Anand, S. P. Potharaju, et al., "AutoMix: Automatically Mixing Language Models," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. arXiv:2310.12963.

[17] I. Ong, A. Almahairi, V. Wu, W.-L. Chiang, T. Wu, J. E. Gonzalez, M. W. Kadous, and I. Stoica, "RouteLLM: Learning to Route LLMs with Preference Data," in *Proc. International Conference on Learning Representations (ICLR)*, 2025. arXiv:2406.18665.

[18] Q. J. Hu, J. Bieker, X. Li, N. Jiang, B. Keigwin, G. Ranganath, K. Keutzer, and S. K. Upadhyay, "RouterBench: A Benchmark for Multi-LLM Routing System," 2024. arXiv:2403.12031.

[19] E. J. Husom, A. Goknil, L. J. Shrestha, M. Bhattarai, et al., "Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency," *ACM Transactions on Internet of Things*, 2025. arXiv:2504.03360.

[20] J. Wei, S. Cao, T. Cao, L. Ma, L. Wang, Y. Zhang, and M. Yang, "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge," in *Proc. European Conference on Computer Systems (EuroSys)*, 2025. arXiv:2407.00088.

[21] Y. Park, J. Hyun, S. Cho, B. Sim, and J. W. Lee, "Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs," in *Proc. International Conference on Machine Learning (ICML)*, 2024 (Oral). arXiv:2402.10517.

[22] P. A. Nair, P. Datta, J. Dean, P. Jain, and A. Kusupati, "Matryoshka Quantization," in *Proc. International Conference on Machine Learning (ICML)*, PMLR vol. 267, pp. 45484–45506, 2025. arXiv:2502.06786.

[23] "MoQE: Improve Quantization Model Performance via Mixture of Quantization Experts," 2025. arXiv:2508.09204. (Preprint; see also OpenReview submission xNF9f5ZszL.)

### Additional works cited in discussion only

[24] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, P. Gao, Y. Qiao, and P. Luo, "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2024 (Spotlight). arXiv:2308.13137.

[25] S. Ma, H. Wang, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, R. Wang, J. Xue, and F. Wei, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," 2024. arXiv:2402.17764.

[26] "Post-Training Quantization of Large Language Models with Microscaling Formats," 2024. arXiv:2405.07135.

[27] "Evaluating the Impact of Post-Training Quantization on Large Language Models for Code Generation," in *Proc. IEEE/ACM International Conference on Program Comprehension (ICPC)*, 2026. arXiv:2503.07103. DOI: 10.1145/3794763.3794810.

[28] H. Li, Y. Zhang, Z. Guo, C. Wang, S. Tang, Q. Zhang, Y. Chen, B. Qi, P. Ye, L. Bai, Z. Wang, and S. Hu, "LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing," in *Findings of the Association for Computational Linguistics: ACL 2026*, pp. 37733–37754, 2026. arXiv:2601.07206.

### Security works (tabulated in rows 24–27, Module VII)

[29] K. Egashira, M. Vero, R. Staab, J. He, and M. Vechev, "Exploiting LLM Quantization," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, 2024. arXiv:2405.18137. DOI: 10.52202/079017-1319.

[30] K. Egashira, R. Staab, M. Vero, J. He, and M. Vechev, "Mind the Gap: A Practical Attack on GGUF Quantization," in *Proc. 42nd International Conference on Machine Learning (ICML)*, PMLR vol. 267, 2025. arXiv:2505.23786.

[31] J. Hong, J. Duan, C. Zhang, Z. Li, C. Xie, K. Lieberman, J. Diffenderfer, B. R. Bartoldson, A. Jaiswal, K. Xu, B. Kailkhura, D. Hendrycks, D. Song, Z. Wang, and B. Li, "Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression," in *Proc. 41st International Conference on Machine Learning (ICML)*, PMLR vol. 235, 2024. arXiv:2403.15447.

[32] A. Shafran, R. Schuster, T. Ristenpart, and V. Shmatikov, "Rerouting LLM Routers," 2025. arXiv:2501.01818.

### Dataset

[33] P. Röttger, H. R. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy, "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models," in *Proc. 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)*, 2024. arXiv:2308.01263.

---

### Verification note

All references above were located and their bibliographic metadata (authors, venue, year,
arXiv identifier) confirmed against arXiv listings, the ACL Anthology, and official
conference proceedings pages during preparation of this survey.

Qualifications that should be resolved before formal submission:

1. Author lists for [4] and [7] are truncated with "et al." where the complete list was
   not individually confirmed; expand them from the respective arXiv pages.
2. Author lists were not obtained for [23], [26] and [27]; these are cited by title,
   venue and identifier only.
3. [23] (MoQE) remains a preprint and is **load-bearing**: it bounds this project's G2
   and G3 novelty claims, so its exact contribution should be read in full rather than
   taken from its abstract before the claims are submitted.
4. [29]–[32] were verified against arXiv and proceedings listings when the security
   module was added. [32] is an arXiv preprint with no confirmed venue. The dataset entry
   for [29] is kept general because the specific models were not confirmed from the
   abstract; check the paper before submission. [33] is used as a benchmark; its
   repository declares no licence, so the prompts are downloaded at build time rather than
   redistributed.

**Correction on record.** An earlier draft attributed to RouterBench [18] the finding
that binary routers such as HybridLLM and FrugalGPT fail to beat the best single model.
That finding belongs to LLMRouterBench [28], not [18]; RouterBench's own abstract claims
only a comparative analysis "highlighting their potentials and limitations". The
attribution has been corrected in Module IV and in G5.

**Provenance of the [28] quotation.** The sentence quoted in Module IV — "several routers
fail to outperform the Best Single, and some (especially binary routers, such as
HybridLLM and FrugalGPT) struggle to trade cost for savings while preserving Best Single
accuracy" — was checked verbatim against the paper body (Section 4.2.2,
"Performance-Cost Settings"). It is **not** in the abstract, which states only that
several recent approaches, including commercial routers, fail to reliably outperform a
simple baseline. Cite the body, not the abstract, when quoting it.

Technical claims and reported figures attributed to each work (for example the 32.39%
mathematical-reasoning degradation in [10], the 1.7% versus 16.0% Japanese evaluation gap
in [9], the 40% reduction in large-model calls in [12], and the 2.61 J/token measurement
in [19]) were drawn from published abstracts and proceedings summaries rather than from
full-text reading of each paper, and are worth spot-checking against the source PDFs
before the figures are quoted in a final report.
