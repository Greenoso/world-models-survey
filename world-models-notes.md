# World Models — Literature Review Notes

Companion to `index.html` (the reveal.js slide deck). A neutral landscape of the world-model literature, assembled to scope the **Tiny World Models for Academic-Scale Training** project. Every numeric figure was verified against the primary source (arXiv / official page); `n/r` = not reported in the primary source. Industrial systems with blog-only disclosure are flagged.

---

## 1. Terminology & lineage

A **world model** is a learned internal model of how an environment evolves: given the current state (and optionally an action) it predicts future **states, observations, and/or rewards**, for use in **planning, control, imagination, or simulation**.

| Year | Source | Contribution |
|---|---|---|
| 1943 | Craik, *The Nature of Explanation* | "Small-scale model" of reality in the mind (conceptual ancestor). |
| 1990 | Schmidhuber, *Making the World Differentiable* (FKI-126-90) | Recurrent "model network" predicts future sensations → backprop a controller through the world. |
| 1991 | Sutton, *Dyna* | Plan by "trying things in your head" with a learned model; interleave model learning + acting. |
| 2018 | Ha & Schmidhuber, *World Models* | **V (VAE) + M (MDN-RNN) + C (controller)**; train a policy entirely inside a "dream". |
| 2022 | LeCun, *A Path Towards AMI* | **JEPA**: predict in latent/representation space; world model as one cognitive module. |
| 2024+ | GameNGen, Genie, Sora | "Neural game engine / interactive generative video / world simulator". |

Terms by community: *"the model"* (model-based RL) · *"world model"* (Dreamer line) · *"JEPA"* (LeCun) · *"neural game engine / world simulator"* (generative video).

---

## 2. Taxonomy axes

- **Prediction space**: pixels (GameNGen, Genie) ↔ latent state (Dreamer, IRIS) ↔ representation, no reconstruction (JEPA, DINO-WM).
- **Purpose**: planning/control (MuZero) ↔ generative simulation (Genie) ↔ representation learning (V-JEPA).
- **Backbone**: RNN/RSSM (PlaNet) ↔ Transformer AR-tokens (IRIS, Genie) ↔ Diffusion (DIAMOND) ↔ SSM/Mamba (emerging).
- **Stochasticity**: deterministic (MuZero dynamics) ↔ stochastic latent-variable (RSSM).
- **Controllability**: action-conditioned (Dreamer, GameNGen) ↔ latent/inferred actions (Genie) ↔ unconditioned (passive video).
- **Data regime**: online, learns while acting (Dreamer, DayDreamer) ↔ offline, fixed corpus (Genie, GameNGen).

---

## 3. Objective-function families

| Family | Loss (sketch) | Examples |
|---|---|---|
| Reconstruction (VAE/ELBO) | reconstruction + KL (+ symlog, KL balancing, free bits) | PlaNet, Dreamer V1–V3 |
| Latent self-prediction (JEPA) | ‖predict − sg(target)‖ in latent space; collapse prevention (EMA/VICReg) | I-JEPA, V-JEPA, DINO-WM |
| Autoregressive token NLL | −log p(next token \| past, action) over VQ tokens | IRIS, TWM, Genie, iVideoGPT |
| Diffusion / denoising | denoise next frame/latent conditioned on past + action | DIAMOND, GameNGen, Oasis, Cosmos |
| Value-equivalent | reward + value + policy (no observation prediction) | MuZero, EfficientZero, TD-MPC2 |

---

## 4. Inputs & outputs

**Inputs** — observations (pixels 64×64→720p, proprioception, depth, multi-camera, text); actions (given discrete/continuous; latent/inferred without labels — Genie; keyboard+mouse — GameNGen/Oasis; text "world events" — Genie-3); a context window of past frames/latents.

**Outputs (= prediction target)** — next observation (RGB at a resolution/FPS); next latent state; reward/value/policy (value-equivalent); future representation (embedding to plan against).

Rule of thumb on cost of the output: reward/value < latent < representation < pixels.

---

## 5. Notable models — Model-based RL

| Model | Year/Venue | Backbone | Params | Objective | Train cost | Inference | Atari100k / headline |
|---|---|---|---|---|---|---|---|
| PlaNet | 2019 ICML | RSSM (GRU200 + 30-d Gauss) | n/r | recon + KL (free nats) + latent overshoot | 1×V100, 10–20h/task | CEM 1000×10 per step | ≈D4PG@100M on DMC, 40–500× less data |
| DreamerV1 | 2020 ICLR | RSSM + CNN | n/r | recon+reward+KL; analytic value grads | 1×V100, ~3h/1M steps | amortized (real-time) | 823 avg, 20 DMC visual @5M |
| DreamerV2 | 2021 ICLR | RSSM 32×32 categorical | WM ~20M | + discount + KL balancing; REINFORCE+λ | 1×V100, <10 days, 200M frames | amortized | Atari-55 median 2.15 (200M) |
| DreamerV3 | 2023 / Nature 2025 | RSSM categorical | 8 / 18 / 37 / 77 / 200M | symlog pred + dyn/rep KL (free bits) | 1×V100; Minecraft 17 GPU-days; Atari200M 16 | amortized | **Atari100k mean 1.12, median 0.49**; 1st diamond |
| DayDreamer | 2022 CoRL | DreamerV2 RSSM 512 | n/r | DreamerV2 losses (on real robots) | 1 GPU async; A1 walks in **1h** | on-robot real-time | quadruped walks from scratch in 1h |
| MuZero | 2020 Nature | ResNet (16 blk) + MCTS | n/r | reward+value+policy+L2 (K=5) | TPUs (8+32 Atari), 1M batches | MCTS 50 (Atari)/800 (board) | Atari-57 mean 4999%, median 2041% (200M) |
| EfficientZero | 2021 NeurIPS | MuZero + SimSiam consistency | n/r (<MuZero) | + self-sup consistency + value-prefix | **4 GPU × 7h/game**, 100k steps | MCTS 50 sims/move | **1st superhuman: mean 1.94, median 1.09** |
| EfficientZero-V2 | 2024 ICML | rep/dyn/pol/val nets | n/r | + Gumbel search (discrete+continuous) | n/r | Gumbel search 32 sims (16 Atari) | **mean 2.43, median 1.29**; beats DreamerV3 50/66 |
| TD-MPC | 2022 ICML | 5 MLPs (TOLD), decoder-free | ~1.5M | consistency+reward+TD value | 1×RTX3090, ~1h/task | MPPI 512×6 per step | SOTA DMC/Meta-World sample-eff. |
| TD-MPC2 | 2024 ICLR | implicit MLP, decoder-free | 1 / 5 / 19 / 48 / 317M | consistency+reward+TD (EMA targets) | 317M: **33 GPU-days on RTX3090** | MPPI H=3, 512 samples | 80 tasks one config; >SAC, >DreamerV3 |
| IRIS | 2023 ICLR | VQ tokenizer (16 tok/frame) + GPT (10L,256) | n/r | recon+commit+perceptual; token CE | **8×A100, ~3.5 GPU-days/game** | amortized | **mean 1.05, median 0.29, 10/26 superhuman** |
| TWM | 2023 ICLR | Transformer-XL (10L,256) + VAE 32×32 | 21.6M | VAE + dyn CE/NLL; actor-critic | **1×A100, ~10h/run** | amortized (4.4M path) | **mean 0.96, median 0.51, 17/26 superhuman** |
| Δ-IRIS | 2024 ICML | delta autoenc (4 tok/frame) + transformer (3L,512) | 25M | delta token CE + reward/term | 1×A100, **~10× faster than IRIS** | amortized | **mean 1.39, IQM 0.65**; SOTA Crafter (16.1) |
| SimPLe | 2019 ICLR | video-pred CNN + LSTM | ~74M | pixel video-pred + reward; PPO in model | 100k steps, 15-iter loop | amortized PPO; 32 ms/frame | beats Rainbow ~13/26 @100k; not human-level |

Atari100k = 26 games, 100k agent steps (400k frames) ≈ 2h human play. 1.0 = human-level. MuZero/DreamerV2 figures are 200M-frame agents, **not** comparable to 100k-budget agents.

---

## 6. Notable models — Generative / neural game engines

| Model | Year/Org | Backbone | Params | Objective | Train compute | Inference | Resolution | Notable |
|---|---|---|---|---|---|---|---|---|
| GameNGen | 2024 Google | SD-1.4 latent diffusion | n/r | diffusion denoising + decoder FT | 128 TPU-v5e, 700k steps, 900M frames | **~20 FPS** (4 steps) / ~50 FPS distilled, 1 TPU-v5 | 320×240 | playable DOOM; PSNR 29.4 |
| Genie | 2024 DeepMind (ICML best) | ST-transformer (tokenizer+LAM+dynamics) | **10.7B** (dyn 10.1B) | latent-action + MaskGIT token pred | 256 TPU-v5p, 942B tokens, ~30k h video | batch | 160×90 (→360p) | controllable worlds from **unlabeled** video |
| Genie-2 | 2024 DeepMind | AR latent diffusion | **blog only** | AR latent diffusion + CFG | not reported | real-time only distilled | n/r | 3D worlds from 1 image, up to ~1 min |
| Genie-3 | 2025 DeepMind | AR (undisclosed) | **blog only** | AR generation | not reported | **24 FPS, real-time** | 720p | minutes-consistent + promptable world events |
| DIAMOND | 2024 NeurIPS | diffusion U-Net (EDM, 3 steps) | Atari 4.4M; CS:GO 381M | EDM diffusion | Atari100k; CS:GO 87h, 1×RTX4090 ×12 d | **~10 FPS** on RTX 3090 | low-res (+upsampler) | Atari100k SOTA **1.46**; CS:GO engine |
| Oasis | 2024 Decart/Etched | ViT autoencoder + DiT (Diffusion Forcing) | 500M (open); demo larger (n/r) | latent diffusion, autoregressive | Minecraft video (scale n/r) | **20 FPS** on H100 | 360p | first open real-time playable diffusion WM |
| iVideoGPT | 2024 NeurIPS | VQGAN + LLaMA-style GPT | 138M / 436M | autoregressive token NLL | ~1.5M trajectories (OXE+SSv2) | n/r | 64²/256² | compressive tokenization; scalable AR WM |
| Cosmos | 2025 NVIDIA | diffusion (7B/14B) + AR (4B/12B) | 4–14B | diffusion + AR NLL | **10,000 H100 × ~3 mo**; 100M clips (20M h raw) | batch (Nano/Super/Ultra) | 720p | open world-foundation-model platform |
| GAIA-1 | 2023 Wayve | AR transformer + video diffusion decoder | 9B (6.5B + 2.6B) | AR token NLL + diffusion decoder | 4,700 h driving; 64+32 A100 × 15 d | batch | n/r | early large driving world model |
| GAIA-2 | 2025 Wayve | space-time transformer (flow matching) | 8.4B (+tok) | flow matching | ~25M 2-s seqs; HW n/r | batch | 448×960 ×5 views, 20–30 Hz | multi-view, richly controllable driving |
| Vista | 2024 NeurIPS | latent video diffusion (SVD-based) | n/r (~1.5B base) | latent diffusion + 2 aux losses | nuScenes + OpenDV; HW n/r | 10 Hz (claimed) | 576×1024 | generalizable driving WM, open weights |
| Sora | 2024 OpenAI | diffusion transformer (spacetime patches) | **not reported** | text-conditional diffusion | **not reported** | batch | variable, up to ~1 min | "world simulator" framing; report only |
| Muse/WHAM | 2025 Microsoft (Nature) | VQ-GAN + transformer (WHAM); MaskGIT (WHAMM) | WHAM 1.6B; WHAMM ~750M | token NLL / MaskGIT | WHAM >1B img-action pairs (~7 y play) | WHAM ~1 FPS; WHAMM **10+ FPS** | 300×180 / 640×360 | game-ideation; AR→MaskGIT route to real-time |

**Disclosure gaps:** Genie-2/3 and Sora reveal no params/data/compute — treat specs as capability claims, not engineering numbers.

---

## 7. Representation / non-reconstructive (JEPA) models

- **I-JEPA** (CVPR 2023, arXiv:2301.08243) — masked-image latent prediction; ViT-H/14 on **16×A100, <72h**.
- **V-JEPA** (2024, arXiv:2404.08471) — feature prediction on ~2M videos; ViT-H/16 → 81.9% Kinetics-400, 72.2% SSv2.
- **V-JEPA 2** (2025, arXiv:2506.09985) — **1.2B** video WM, >1M h video; V-JEPA 2-AC post-trained on **<62h** robot video → **zero-shot** pick-and-place planning.
- **DINO-WM** (ICML 2025, arXiv:2411.04983) — world model in **frozen DINOv2** patch space; task-agnostic zero-shot planning via latent rollouts. Total params / GPU-hours n/r.
- Related 2025: *DINO as a Foundation for Video World Models* (arXiv:2507.19468); *Improving Transformer World Models for Data-Efficient RL* (arXiv:2502.01591).

---

## 8. Benchmarks

- **Atari 100k** — 26 games, 100k steps (~2h play); canonical sample-efficiency / academic-scale yardstick.
- **DM Control Suite** — continuous control from state or pixels, 0–1000/task.
- **Crafter** — 2D survival, 22 achievements; broad capability, cheap.
- **Minecraft (diamond)** — open-ended, long-horizon, sparse reward.
- **Procgen** — 16 procedural games; isolates generalization (train→test gap).
- **Generative video** — FID (per-frame), **FVD** (Fréchet Video Distance over I3D spatiotemporal features; lower = better), action-controllability, long-horizon consistency.

---

## 9. Cost spectrum (verified)

| Model / class | Params | Training compute | Data | Inference |
|---|---|---|---|---|
| Ha & Schmidhuber WM | small (VAE+RNN+linear) | single GPU, hours (n/r) | self-collected rollouts | tiny |
| IRIS | n/r | 8×A100, ~3.5 GPU-days/game | Atari100k | single GPU |
| DIAMOND (CS:GO) | 381M | 1×RTX4090, 12 days (~288 GPU-h) | 87h gameplay | ~10 FPS / 3090 |
| DreamerV3 (Minecraft) | up to 200M | 1×V100, 17 GPU-days (~408 GPU-h) | online | single GPU |
| GAIA-1 | 9B | 96×A100 × 15 d (~3.5×10⁴ GPU-h) | 4,700h driving | batch |
| Genie | 10.7B | 256 TPU-v5p, 942B tokens (wall-clock n/r) | ~30k h video | batch |
| Cosmos | 4–14B | 10,000 H100 × ~3 mo (~2×10⁷ GPU-h) | 100M clips | batch (tiered) |
| Sora | n/r | n/r | n/r | batch |

≈ **6 orders of magnitude** in training compute, ≈ **4** in parameters. TPU/GPU not directly comparable; several wall-clocks n/r.

Three inference regimes: **amortized policy** (real-time, near-free — Dreamer/IRIS) · **search/planning** (heavy per step — MuZero MCTS, TD-MPC MPPI, PlaNet CEM) · **generative rollout** (per-frame diffusion/AR — real-time only with few steps/distillation). *Cheap to train ≠ cheap to run.*

---

## 10. Efficiency levers for "tiny" world models

1. **Latent-space dynamics** (RSSM) instead of pixel prediction — Dreamer.
2. **Discrete tokens + small transformer** — IRIS; Δ-IRIS (4 tokens/frame, ~10× faster).
3. **No reconstruction** (latent prediction, drop decoder) — JEPA, DINO-WM.
4. **Imagination + replay** for sample efficiency — Atari100k regime.
5. **Self-supervised / consistency aux losses** — EfficientZero.
6. **Distillation · synthetic data · curriculum** — Cosmos-Nano, GameNGen distillation.
7. **State-space backbones (S4 / Mamba)** — near-linear long-context rollouts; under-explored for WMs.

---

## 11. Synthesis (neutral)

- The strongest **sample-efficiency** results (superhuman Atari100k, Minecraft diamond, real-robot planning) come from **sub-200M-parameter, single-GPU** models — capability is not monopolized by scale.
- The genuine cost gap is **visual fidelity and open-domain generality** (photorealistic interactive video), which is exactly what costs 10⁴–10⁷ GPU-hours and proprietary corpora.
- Most promising tiny levers: **no-reconstruction (JEPA)**, **context-aware tokenization (Δ-IRIS)**, **SSM backbones**, and **distillation from large neural engines**.

---

## 12. Full reference list (verified primary sources)

1. Ha & Schmidhuber. World Models. NeurIPS 2018. arXiv:1803.10122
2. Schmidhuber. Making the World Differentiable. FKI-126-90, TU München, 1990.
3. Sutton. Dyna, an Integrated Architecture for Learning, Planning and Reacting. ACM SIGART Bulletin 2(4), 1991.
4. LeCun. A Path Towards Autonomous Machine Intelligence. 2022.
5. Hafner et al. Learning Latent Dynamics for Planning from Pixels (PlaNet). ICML 2019. arXiv:1811.04551
6. Hafner et al. Dream to Control (DreamerV1). ICLR 2020. arXiv:1912.01603
7. Hafner et al. Mastering Atari with Discrete World Models (DreamerV2). ICLR 2021. arXiv:2010.02193
8. Hafner et al. Mastering Diverse Domains through World Models (DreamerV3). Nature 2025. arXiv:2301.04104
9. Wu et al. DayDreamer: World Models for Physical Robot Learning. CoRL 2022. arXiv:2206.14176
10. Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero). Nature 2020. arXiv:1911.08265
11. Ye et al. Mastering Atari Games with Limited Data (EfficientZero). NeurIPS 2021. arXiv:2111.00210
12. Wang et al. EfficientZero V2. ICML 2024. arXiv:2403.00564
13. Hansen et al. Temporal Difference Learning for Model Predictive Control (TD-MPC). ICML 2022. arXiv:2203.04955
14. Hansen et al. TD-MPC2. ICLR 2024. arXiv:2310.16828
15. Micheli et al. Transformers are Sample-Efficient World Models (IRIS). ICLR 2023. arXiv:2209.00588
16. Robine et al. Transformer-based World Models Are Happy With 100k Interactions (TWM). ICLR 2023. arXiv:2303.07109
17. Micheli et al. Efficient World Models with Context-Aware Tokenization (Δ-IRIS). ICML 2024. arXiv:2406.19320
18. Kaiser et al. Model-Based RL for Atari (SimPLe). ICLR 2020. arXiv:1903.00374
19. Valevski et al. Diffusion Models Are Real-Time Game Engines (GameNGen). 2024. arXiv:2408.14837
20. Bruce et al. Genie: Generative Interactive Environments. ICML 2024. arXiv:2402.15391
21. Google DeepMind. Genie 2 (blog, Dec 2024); Genie 3 (blog, Aug 2025).
22. Alonso et al. Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND). NeurIPS 2024. arXiv:2405.12399
23. Decart & Etched. Oasis: A Universe in a Transformer. 2024. github.com/etched-ai/open-oasis
24. Wu et al. iVideoGPT: Interactive VideoGPTs are Scalable World Models. NeurIPS 2024. arXiv:2405.15223
25. NVIDIA. Cosmos World Foundation Model Platform for Physical AI. 2025. arXiv:2501.03575
26. Hu et al. (Wayve). GAIA-1. 2023. arXiv:2309.17080
27. Wayve. GAIA-2. 2025. arXiv:2503.20523
28. Gao et al. Vista. NeurIPS 2024. arXiv:2405.17398
29. OpenAI. Video generation models as world simulators (Sora). 2024.
30. Microsoft Research. Muse / WHAM. Nature 2025.
31. Assran et al. I-JEPA. CVPR 2023. arXiv:2301.08243
32. Bardes et al. V-JEPA. 2024. arXiv:2404.08471
33. Assran et al. V-JEPA 2. 2025. arXiv:2506.09985
34. Zhou et al. DINO-WM. ICML 2025. arXiv:2411.04983
35. Cobbe et al. Procgen. ICML 2020. arXiv:1912.01588
36. Hafner. Crafter. 2021. arXiv:2109.06780
37. Unterthiner et al. FVD: Towards Accurate Generative Models of Video. 2018.
