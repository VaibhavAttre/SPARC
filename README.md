# Cache Replacement Optimization

**CS 175 — Reinforcement Learning Project**  
**Team:** Vaibhav Attre and Calvin Zhang

> **Research question:** Can a reinforcement learning agent jointly adapt **cache replacement** and **cache write** policies to reduce simulated I/O latency and storage-write traffic as workloads change?

## 1. Project Overview

A cache keeps frequently or recently requested data close to the program using it. When the cache is full, a **replacement policy** decides which entry to evict. When a program writes data, a **write policy** decides when that change reaches the backing store.

Different workloads favor different policies. An access stream with strong recency may benefit from LRU; a stable set of popular keys may benefit from LFU. Write behavior adds another trade-off: write-through propagates every write promptly, while write-back delays backing-store writes but may require costly flushes of dirty blocks.

**Our project combines these decisions.** We will build a trace-driven cache simulator with three replacement policies and two write policies, then train tabular Q-learning and Deep Q-learning agents to choose policy combinations dynamically. We will evaluate them against fixed-policy and adaptive baselines on synthetic workloads with changing access and write patterns.

### Goals

- Implement **LRU**, **LFU**, and **FIFO** replacement in a single persistent cache.
- Implement **write-through** and **write-back**, including dirty-block tracking and flush costs.
- Train **tabular Q-learning** and **Deep Q-learning (DQN)** agents to choose among the six policy combinations.
- Measure **cache hit rate**, **simulated I/O latency**, **backing-store write traffic**, **policy-switching cost**, and **decision overhead**.
- Test whether learning-based joint adaptation beats fixed or simpler adaptive baselines on previously unseen workload traces.

**Scope:** This is a **software simulation**, not a modification to a CPU's hardware cache controller. Report estimated latency from an explicit cost model separately from measured Python execution time.

---

## 2. The Policy Space

### Cache replacement policies

| Policy | Eviction rule | Intuition |
|---|---|---|
| **LRU** (Least Recently Used) | Evict the entry accessed least recently. | Often effective with temporal locality. |
| **LFU** (Least Frequently Used) | Evict the entry with the lowest access count. | Often effective with stable popularity skew. |
| **FIFO** (First In, First Out) | Evict the entry inserted earliest. | Simple baseline; ignores access frequency and recency. |

Tie-breaking behavior will be deterministic and documented. We will maintain the information required by **all three policies at every request**, regardless of the currently active policy. The cache must **not reset** when the agent changes policies.

### Cache write policies

| Policy | On a write | Trade-off |
|---|---|---|
| **Write-through (WT)** | Update the cached entry and write the change to the backing store immediately. | Simpler persistence behavior, but more storage-write traffic. |
| **Write-back (WB)** | Update the cached entry and mark it **dirty**; flush it later when necessary. | Can combine repeated writes, but dirty evictions and flushes cost I/O. |

A **dirty block** has modifications that have not yet been propagated to the backing store. A **clean block** does not require a write-back before eviction.

### Six possible RL actions

The agent chooses one pair each decision window:

| Action ID | Replacement policy | Write policy |
|---|---|---|
| 0 | LRU | Write-through |
| 1 | LRU | Write-back |
| 2 | LFU | Write-through |
| 3 | LFU | Write-back |
| 4 | FIFO | Write-through |
| 5 | FIFO | Write-back |

The agent **selects a policy combination**, not an individual cache entry. The selected replacement policy determines which entry is evicted when eviction is needed.

---

## 3. System Architecture

```text
                 Synthetic / trace-based workload
                 (READ/WRITE requests, keys, phases)
                                |
                                v
                    +-----------------------+
                    |   Cache Simulator     |
                    |  - persistent entries |
                    |  - LRU/LFU/FIFO data  |
                    |  - dirty-bit tracking |
                    |  - latency accounting |
                    +-----------------------+
                         |             ^
                   state/reward        | chosen action
                         v             |
                    +-----------------------+
                    | Gymnasium Environment |
                    |  K requests per step  |
                    +-----------------------+
                                ^
                                |
                    +-----------------------+
                    |   RL Agent            |
                    |  Tabular Q / DQN      |
                    +-----------------------+
                                |
                                v
                  Metrics, plots, and comparisons
```

### One decision window

1. Observe recent cache and workload statistics.
2. Choose **one of the six replacement/write-policy combinations**.
3. Apply that combination without clearing the cache or its bookkeeping.
4. Process the next `K` requests.
5. Accumulate hits, misses, storage reads/writes, simulated latency, dirty evictions, and switching costs.
6. Compute the reward, construct the next state, and update the agent.
7. Repeat until the trace ends.

We will initially use `K = 1,000` as an **experiment parameter**, not a fixed requirement. We will test multiple window sizes.

---

## 4. Cache Simulator Design

### Request format

Each request should include at least:

```text
request_id,timestamp,op,key,size,phase
0,0.000,R,17,1,hot-set-A
1,0.002,W,42,1,hot-set-A
2,0.005,R,17,1,hot-set-A
3,0.006,W,17,1,hot-set-A
```

- `request_id`: position in the trace. Acts as the **logical clock** and deterministic tie-breaker for LRU/FIFO/LFU.
- `timestamp`: non-decreasing arrival time in simulated time units (fixed spacing or Poisson arrivals for synthetic traces). Kept separate from the cost-model latency, which never rewrites it.
- `op`: `R` (read) or `W` (write)
- `key`: logical block/object identifier
- `size`: object size (start with fixed-size blocks, `size = 1`; variable size is optional)
- `phase`: optional label used for analysis, **never passed to the agent** (strip it before the environment builds observations)

### Per-cache-entry metadata

Maintain at least:

- Object/block key and cached data version (or a lightweight correctness marker)
- Insertion order or insertion time (FIFO), as a logical `request_id` value
- Last-access time (LRU), as a logical `request_id` value
- Access count (LFU)
- Dirty flag (write-back)

### Request behavior

| Event | Required simulator behavior |
|---|---|
| **Read hit** | Serve from cache; update recency/frequency metadata. |
| **Read miss** | Fetch from backing store; if full, select and evict a victim; insert fetched entry. |
| **Write hit, WT** | Update cache and backing store immediately; entry remains clean. |
| **Write hit, WB** | Update cache and mark entry dirty; defer backing-store write. |
| **Write miss** | Use one consistent write-miss allocation rule (see below), evicting a victim if needed. |
| **Clean eviction** | Remove the block without a backing-store write. |
| **Dirty eviction** | Write the modified block to backing store and count its latency/traffic before removing it. |

**Recommended MVP assumption:** Use **write-allocate** for both WT and WB on write misses. A write miss loads/allocates the block, then applies the write using the selected policy. Keep this rule constant across all experiments so the agent is only optimizing the two policy dimensions stated in the presentation. Document the backing-store read cost associated with write allocation.

### Switching between write policies

Changing replacement policy must not discard cached entries or reset metadata. Changing write policy needs additional care:

- **WB → WT:** Flush all dirty resident blocks to backing store, charge the corresponding I/O cost, and mark them clean before WT is active.
- **WT → WB:** Retain current cache contents. Subsequent writes can create dirty blocks.
- **No change:** Do not charge an artificial transition cost beyond any explicitly modeled decision overhead.

At the **end of a trace**, flush remaining dirty blocks and include these writes in the final totals (or report outstanding dirty data separately). This prevents write-back from appearing artificially cheap simply because the experiment stops before its writes occur.

### Simulator counters

At minimum, expose:

```text
requests, reads, writes
hits, misses, evictions, dirty_evictions
backing_store_reads, backing_store_writes
bytes_read, bytes_written
estimated_io_latency
replacement_switches, write_policy_switches
write_policy_flushes
```

### Simulated I/O cost model

Start with a simple, explicit set of **configurable** costs:

```text
cache_hit_cost       = configurable
backing_read_cost    = configurable
backing_write_cost   = configurable
switch_overhead_cost = configurable
```

Compute total estimated latency by summing the events incurred. These are **simulation units**, not measured SSD or hardware latency. You can later add object-size-dependent cost or a more realistic storage model as long as the baseline comparisons use identical assumptions.

---

## 5. Reinforcement Learning Formulation

We model cache policy control as a **Markov decision process approximation** in which each environment step processes `K` requests.

### State / observation

The presentation proposes four main state features, computed over the previous window of `K` requests:

| Feature | Example measurement |
|---|---|
| **Cache hit rate** | Hits divided by requests in the previous window. |
| **Read/write ratio** | Fraction of requests that are writes (or a bounded read/write ratio). |
| **Access locality** | Fraction of requests whose key appeared in the previous `N` requests (cheaper than mean reuse distance). |
| **Dirty-page ratio** | Number of dirty cached blocks divided by current cache occupancy. |

**Required additional features:**

- **Current replacement and write policies.** Otherwise, the agent cannot tell whether an action incurs a switching cost.
- **One skew/scan feature:** the unique-key ratio in the window (flags sequential scans) or the share of requests going to the top-k keys (flags Zipfian skew). The four core features alone barely distinguish LRU, LFU, and FIFO workloads.

**Notes:**

- Normalize every feature to [0, 1].
- Tabular Q-learning needs a small state space: use about 5 features with 3-4 bins each. Six features at 5 bins each, times six current-policy combinations, is roughly 94k states, which is too many to learn from. DQN can use the full feature set.
- Request rate (mean inter-arrival time, from `timestamp`) is an optional later feature. It matters little with fixed `K`-request windows.
- Avoid giving the agent future workload phase labels or other information unavailable at decision time.

### Action

A discrete choice from `0` through `5`, representing the six combinations in Section 2.

### Reward

The presentation prioritizes **lower I/O latency**, **fewer storage writes**, and **less switching overhead**, while also considering cache hits. A starting reward can be:

\[
R_t = w_h \cdot \mathrm{hit\_rate}_t
- w_l \cdot \mathrm{normalized\_latency}_t
- w_w \cdot \mathrm{normalized\_storage\_writes}_t
- w_s \cdot \mathrm{switch\_cost}_t
\]

The exact weights and normalization are **design parameters to tune**. Avoid adding raw counts measured on different scales without normalization. Include all costs generated during the window, including flushes caused by switching write policies.

### Episode

An episode is a complete workload trace (or a fixed-length portion of one). The cache is reset **between episodes**, not every time the selected policy changes.

### Agent A: Tabular Q-learning

- Discretize continuous features into a modest number of bins.
- Use epsilon-greedy exploration with decaying epsilon.
- Update the Q-table after every decision window.
- Begin with a small state space and tune learning rate, discount factor, epsilon decay, and reward weights.

### Agent B: Deep Q-learning (DQN)

- Feed normalized continuous features to a small neural network.
- Output six Q-values, one per policy combination.
- Use an experience replay buffer and a target network for stability.
- Compare DQN with tabular Q-learning **using the same simulator, trace splits, and reward definition**.

Build and validate tabular Q-learning first, then implement DQN.

---

## 6. Workloads and Data

The presentation focuses on changing access patterns and read/write behavior. We will initially generate our own **synthetic traces** so that we can control the sources of variation.

| Workload | What changes | Why it matters |
|---|---|---|
| **Zipfian popularity** | A small set of objects receives a large fraction of accesses. | Tests sensitivity to frequency skew. |
| **Temporal locality** | Recently accessed objects are likely to be accessed again. | Tests recency-sensitive policies. |
| **Sequential scan** | Many distinct keys are accessed once in sequence. | Tests scan resistance and cache pollution. |
| **Hot-set shift** | The popular object set changes partway through the trace. | Tests replacement-policy adaptation. |
| **Read-heavy → write-heavy** | Write fraction increases over time. | Tests adaptation of write policy. |
| **Repeated writes → dispersed writes** | Writes target either a few blocks repeatedly or many different blocks. | Tests storage-write coalescing under WB. |
| **Mixed phases** | Locality, popularity, and read/write mix change together. | Tests joint adaptation under nonstationarity. |

For each generator, expose a random seed, trace length, number of unique keys, cache capacity, write fraction, and workload-specific parameters. Create **train**, **validation**, and **test** splits with distinct seeds and, where possible, unseen phase combinations.

**Important:** Do not assume LFU always wins a Zipfian trace or that WB always wins a write-heavy trace. Measure which policies actually perform best under the chosen cache size and cost model.

Optional future work: evaluate on external request traces if they include (or can reliably provide) the read/write events required by the model. Traces with only object IDs cannot directly validate the write-policy part of the project.

---

## 7. Evaluation Plan

### Baselines

1. **All six fixed combinations:** LRU/LFU/FIFO × WT/WB.
2. **Best fixed policy (oracle in hindsight):** Best of the six *on each evaluation trace*, clearly labeled as an offline upper-reference baseline, not a deployable controller.
3. **Random switcher:** Select among the six combinations using the same decision interval.
4. **Tabular Q-learning:** Learned adaptive controller.
5. **DQN:** Learned adaptive controller with continuous-valued features.

Optional if time permits: a simple heuristic controller, or an existing adaptive replacement algorithm for a replacement-only comparison. Note that replacement-only methods do not directly solve the joint write-policy problem.

### Primary metrics

| Metric | Definition / interpretation |
|---|---|
| **Cache hit rate** | `hits / total_requests` (higher is better). |
| **Simulated I/O latency** | Total or mean cost from the documented storage/cache model (lower is better). |
| **Backing-store writes** | Number of propagated writes, including dirty flushes (lower is generally better). |
| **Write traffic** | Total bytes written to backing store. |
| **Dirty evictions / flushes** | Extra I/O caused by delayed writes. |
| **Switch count and switch cost** | How frequently the controller changes either policy and what those changes cost. |
| **Decision overhead** | Measured policy-selection CPU time, reported separately from simulated I/O latency. |

### Experiments

- **Fixed vs adaptive:** Does joint RL adaptation outperform all six fixed choices?
- **Tabular vs DQN:** Does the larger model improve unseen-trace performance enough to justify its overhead?
- **Changing workloads:** How quickly does each agent respond after an access or write-pattern shift?
- **Ablation — replacement only:** Hold write policy constant while learning LRU/LFU/FIFO.
- **Ablation — write only:** Hold replacement policy constant while learning WT/WB.
- **Joint vs separate:** Does learning both dimensions together beat adapting only one?
- **Decision interval:** Sweep `K` (for example, 100, 500, 1,000, 5,000 requests).
- **Reward sensitivity:** Vary the penalties for latency, storage writes, and switching.
- **State sensitivity:** Remove or add features to test which information helps.
- **Cache capacity:** Repeat across multiple cache-to-workload-size ratios.

Use multiple random seeds, display mean and variability, and keep all comparisons on identical request sequences and cost assumptions. Never train on evaluation traces used for the final reported results.

### Suggested plots

- Hit rate and estimated I/O latency by baseline/controller.
- Total storage writes and write traffic by controller.
- Learning curves (episode reward vs training episode).
- Active replacement/write policy over time, annotated with true workload shifts **for visualization only**.
- Policy-switching frequency and decision overhead.
- Ablation results for replacement-only, write-only, and joint adaptation.

---

## 8. Suggested Repository Structure

The following is a **proposed layout**; files and commands are implementation targets, not existing scripts.

```text
cache-replacement-optimization/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── cache/
│   │   ├── simulator.py          # persistent cache, reads/writes, dirty blocks
│   │   ├── replacement.py        # LRU, LFU, FIFO victim selection
│   │   ├── write_policy.py       # WT, WB, flushing and switching
│   │   └── cost_model.py         # latency and I/O accounting
│   ├── workloads/
│   │   ├── generators.py         # synthetic access/RW patterns
│   │   └── traces.py             # trace loading and validation
│   ├── rl/
│   │   ├── environment.py        # Gymnasium wrapper
│   │   ├── tabular_q.py          # epsilon-greedy Q-learning
│   │   └── dqn.py                # Deep Q-learning
│   ├── evaluation/
│   │   ├── baselines.py          # 6 fixed policies, random, oracle
│   │   ├── metrics.py            # aggregate evaluation counters
│   │   └── plots.py              # figures
│   └── utils/
│       └── seeding.py
├── scripts/
│   ├── generate_traces.py
│   ├── run_baselines.py
│   ├── train_tabular.py
│   ├── train_dqn.py
│   └── evaluate.py
├── tests/
│   ├── test_replacement.py
│   ├── test_write_policies.py
│   ├── test_policy_switching.py
│   └── test_environment.py
├── data/                       # generated traces (optionally gitignored)
└── results/                    # tables, checkpoints, plots
```

### Tools

- **Python 3.11+**
- **NumPy / pandas** — data generation, processing, and metrics
- **Gymnasium** — RL environment interface
- **PyTorch** — DQN
- **Matplotlib** — charts
- **pytest** — correctness tests

**libCacheSim:** If desired, use it to sanity-check fixed LRU/LFU/FIFO results. The unified simulator still needs explicit support for read/write requests, dirty blocks, flush costs, and switching among the six combinations; do not assume a replacement-only API implements all of this automatically.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy pandas gymnasium torch matplotlib pytest
```

Once `requirements.txt` has been created, use `pip install -r requirements.txt` instead.

---

## 9. Implementation Roadmap

### Milestone 1 — Simulator correctness

- [ ] Define request schema, cache capacity, and fixed write-allocate semantics.
- [ ] Implement hit/miss handling, eviction, and cache metadata.
- [ ] Implement and unit-test LRU, LFU, and FIFO.
- [ ] Implement and unit-test write-through and write-back.
- [ ] Verify dirty eviction, WB→WT flushing, and end-of-trace flushing.
- [ ] Add configurable latency/storage-write counters.

**Done when:** Tiny hand-calculated traces produce the expected cache contents, victim choices, storage writes, and latencies.

### Milestone 2 — Workload generators and fixed baselines

- [ ] Generate deterministic read/write traces with different locality patterns.
- [ ] Generate workloads with shifting hot sets and changing write fractions.
- [ ] Run all six fixed policy combinations on identical traces.
- [ ] Export baseline comparison tables and plots.

**Done when:** We can identify which fixed combinations perform best on several distinct traces.

### Milestone 3 — Gymnasium + tabular Q-learning

- [ ] Wrap the persistent simulator in a Gymnasium environment.
- [ ] Implement the four core state features and current-policy identity.
- [ ] Implement six actions and a correctly scaled reward.
- [ ] Add windowed decisions, episode reset, and training/evaluation modes.
- [ ] Implement tabular Q-learning with epsilon-greedy exploration.
- [ ] Train and plot reward curves; compare against fixed and random baselines.

**Done when:** The agent learns a nonrandom policy and can be evaluated reproducibly on held-out traces.

### Milestone 4 — DQN and performance experiments

- [ ] Implement the DQN agent, replay buffer, and target network.
- [ ] Evaluate tabular Q-learning vs DQN on the same held-out traces.
- [ ] Measure latency, storage writes, hits, switching, and decision time.
- [ ] Run decision-window, reward-weight, and state-feature ablations.
- [ ] Run replacement-only vs write-only vs joint-adaptation comparisons.
- [ ] Aggregate multiple seeds and generate final plots/tables.

**Done when:** Results answer whether joint adaptation improves performance and whether RL overhead is worthwhile.

---

## 10. Correctness and Fairness Checklist

Before trusting results, verify:

- [ ] Every policy sees exactly the **same** test trace and cache capacity.
- [ ] A policy switch does **not** reset cache contents or policy metadata.
- [ ] Hit/miss counts match a hand-checked trace.
- [ ] WT writes reach backing store immediately and leave no dirty entry.
- [ ] WB repeated writes can be coalesced before the eventual flush.
- [ ] Dirty evictions and WB→WT transitions charge the correct write cost.
- [ ] All pending dirty data is accounted for at episode end.
- [ ] All approaches use the same write-miss allocation and cost model.
- [ ] RL agents see only observations available up to the decision point.
- [ ] Model selection uses validation traces, not the final test traces.
- [ ] Simulated I/O latency is distinguished from Python execution time.
- [ ] Results include multi-seed variance, not only a single best run.

---

## 11. Team Responsibilities

The presentation identifies **Vaibhav Attre** and **Calvin Zhang** as the project team. The following working division reflects the **earlier written proposal** and can be adjusted now that write policies are core to the project:

| Vaibhav | Calvin |
|---|---|
| Cache simulator; LRU/LFU/FIFO | Gymnasium environment |
| Write-through/write-back semantics and I/O cost model | Tabular Q-learning and DQN |
| Fixed and random baselines | State/reward design and hyperparameter tuning |
| Workload generators and evaluation metrics | Training and validation experiments |
| Correctness tests and performance plots | Results analysis and comparative experiments |

Both members should review the shared simulator/RL interface and run the final experiments.

---

## 12. Expected Deliverables

1. A tested simulator that supports three replacement policies and two write policies.
2. Deterministic synthetic workload generators with changing access and write phases.
3. Tabular Q-learning and DQN agents selecting among six policy combinations.
4. A fair evaluation pipeline with fixed, random, and learned-policy baselines.
5. Figures and tables reporting hit rate, simulated I/O latency, storage writes, switching costs, and agent overhead.
6. A final CS 175 write-up discussing where joint adaptation helps, where it fails, and the limitations of a synthetic simulator.

**Success does not require RL to beat every baseline.** A rigorous negative result—explaining when adaptation costs outweigh benefits—is also a useful finding.

---

## 13. Related Research

The project presentation discusses prior work on adaptive replacement and learning-based cache control:

- **LeCaR (2018):** Vietri, Giuseppe, et al. *Driving Cache Replacement with ML-based LeCaR.* 10th USENIX Workshop on Hot Topics in Storage and File Systems (HotStorage '18). [Paper](https://www.usenix.org/conference/hotstorage18/presentation/vietri)
- **CACHEUS (2021):** Rodriguez, Liana V., et al. *Learning Cache Replacement with CACHEUS.* 19th USENIX Conference on File and Storage Technologies (FAST '21), pp. 341–354. [Paper](https://www.usenix.org/conference/fast21/presentation/rodriguez)
- **Deep RL cache replacement (2022):** Zhou, Yang, et al. *An End-to-End Automatic Cache Replacement Policy Using Deep Reinforcement Learning.* Proceedings of ICAPS, vol. 32, no. 1, pp. 537–545. [Paper](https://doi.org/10.1609/icaps.v32i1.19840)

**Proposed distinguishing direction:** Study **joint** control of replacement *and* write policy using tabular and deep Q-learning under changing workloads, rather than adjusting replacement alone. This is the project's intended research contribution, **not a verified claim of uniqueness over all existing literature**.

---

## 14. Getting Started — First Work Session

To avoid building the RL code on top of an incorrect simulator, do the following **in order**:

1. Create the repository structure and agree on the request schema (`R`/`W`, key, phase).
2. Choose a small test cache (e.g., **capacity 3**) and write a short hand-worked trace.
3. Implement **LRU + WT** first; verify hits, misses, and immediate writes.
4. Add **WB**; verify dirty flags, dirty evictions, flush costs, and the final flush.
5. Add LFU and FIFO while keeping one shared cache state and all policy metadata.
6. Only after correctness tests pass, generate longer traces and compare all **six fixed combinations**.
7. Then hand the tested simulator interface to the Gymnasium/RL implementation.

**First milestone:** A correct cache simulator where `LRU × WT` and `LRU × WB` run on the same short read/write trace and produce explainable, different storage-write counts.

---

*README based on the team's 12-slide “Cache Replacement Optimization” presentation. Implementation details (e.g., request schema, write allocation, reward weights, cost constants, proposed directory layout, and milestone order) are concrete recommendations to make the presentation's design executable.*
