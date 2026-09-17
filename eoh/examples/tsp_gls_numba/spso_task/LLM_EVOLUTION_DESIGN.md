# LLM-generated module evolution: implementation contract

This document specifies the requested online S-PSO experiment. The existing
set-valued position, possibility-valued velocity, alpha cut, five TSP GLS
slots (`E/F/H/T/W`), and original EoH evaluation entry point remain the
foundation. Generation 0 is initialization; generations 1..G use PSO.

## Configuration and experiment identity

- Expose `MODULES_PER_SLOT=10`, `INITIAL_SAMPLES=20`, `POPULATION_SIZE=10`,
  `HEURISTIC_OPERATORS=["p1", "p2", "p3", "p4"]`, and
  `USE_MODULE_DESCRIPTIONS=True` in `runSPSO.py`. The operator list is ordered,
  nonempty, duplicate-free, and contains only `p0`..`p4`. Do not reinterpret
  the list as weights for choosing one operator: **run every listed operator
  once for each parent in every evolution generation**. `["p0"]` produces N
  offspring; `["p1", "p2"]` produces 2N; the default produces 4N.
- `p0` labels every generation-0 sample, independent of the later operator
  list. Record the exact ordered list, description flag, LLM model, seed,
  library version/hash, and evaluation settings in run metadata/checkpoints.
- A full run requests `INITIAL_SAMPLES + GENERATIONS * POPULATION_SIZE *
  len(HEURISTIC_OPERATORS)` candidate evaluations (820 for 20+20*10*4).
  Count attempted, valid, failed, cached, and independently evaluated samples
  separately. A budget must not silently truncate an operator batch.

## Typed, versioned module library

- Online initialization asks the LLM to propose **exactly ten validated,
  distinct modules per slot**. Existing TSP modules may be shown as examples,
  but do not silently count pre-existing fixed modules as LLM-generated ones.
  Each accepted module has a stable slot-local ID, a one-sentence description,
  bounded parameter schema, an executable typed implementation, generation,
  source operator, and content hash. Save the accepted library and rejected
  proposals with reasons under the run output. Restore the library *before*
  restoring any position/velocity/checkpoint.
- Enforce a two-step LLM protocol for each code-producing proposal: first
  request and validate the one-sentence design/description, then send that
  description back and request the implementation. Preserve both prompt and
  response for audit. A single JSON response containing fields in a given
  order does not demonstrate this temporal order.
- Store implementations as a small, typed TSP recipe/DSL. The TSP adapter
  renders recipes into actual Python snippets for the existing `E` candidate
  edge list, `F` nonnegative edge feature, `H` nonnegative history factor,
  `T` nonnegative score transform, and `W` symmetric write-back roles. The
  recipe must be capable of new parameterized combinations of safe primitives,
  not just renamed copies. Validate recipe schema, parameter bounds, slot
  membership, uniqueness by normalized implementation+parameters, and a
  small contract execution before accepting it. Never execute arbitrary raw
  LLM Python to register a module. Send rendered snippet code to the LLM in
  later cut-set prompts. Preserve the public three-argument
  `update_edge_distance(edge_distance, local_opt_tour, edge_n_used)` signature;
  the output must be a finite, nonnegative, symmetric n-by-n matrix without
  mutating inputs. Keep the original `TSPGLSWithDetails` / EoH GLS evaluator.
- Keep `SetPosition` capacity K=3 per slot. Each decoded `ParticlePosition`
  must choose exactly one module per slot; its chosen module must belong to
  that particle's `SetPosition`. Newly accepted modules from p1/p2/p3 may
  replace one member of the child's set to preserve K. Extend the child's
  velocity with zero for each new library member.
- `USE_MODULE_DESCRIPTIONS=False` must omit module descriptions from **all**
  selection/operator prompts, including current/pbest/gbest text fields that
  might reveal them. Keep descriptions in the saved library and logs. Allow
  loading the same frozen library for on/off ablations so only the prompt
  condition changes. Never switch to the old random library without labeling
  the run `legacy/offline`.

## Generation 0 (`p0`)

1. Build/save the 10-per-slot library once. Use twenty independent p0 LLM
   requests over that library to generate twenty legal, diverse decoded
   sequences, one module and validated parameters per slot. Require at least
   one tunable parameter in each p0 sequence if p4 is enabled. Validate and
   retry malformed or duplicate responses with a bounded retry limit; fail
   clearly if twenty legal examples cannot be obtained.
2. For each decoded sequence, construct a capacity-three `SetPosition` that
   contains all five selected modules and fills the other places from the
   same registered library using seeded randomness. Initialize all velocity
   possibilities to zero, compile and evaluate all twenty with the current
   EoH GLS evaluation path, record each with `operator="p0"` and no parent.
3. Keep the ten best **distinct positions** by objective. Do not deduplicate
   merely because two positions have equal fitness. Set each particle's
   pbest to itself and gbest to the best. Fail if fewer than ten valid initial
   candidates remain.

## Evolution generation 1..G

1. Freeze population, pbest, gbest, module library, and RNG-derived seeds.
   For each parent, update its set-valued velocity once, compute the alpha cut
   once, and construct one capacity-three candidate set per slot once.
2. For each parent, make one p0 **base sequence** using only that parent's
   constructed cut sets and their rendered module code. Include descriptions
   only when the flag is on. p0 must not consult the old parent sequence.
   Reuse this same base sequence for every configured operator; a listed `p0`
   yields the base as its offspring without another operator call.
3. For each `(parent, operator)` in stable parent/list order, create one child:
   - `p1`: use the old parent sequence and the base sequence; propose a new
     *similar* module in at least one corresponding slot, first its one-line
     description and then its typed implementation. Similarity must be checked
     against a declared semantic family/recipe relation, not only asserted in
     prose. Other slots may keep base choices.
   - `p2`: same two reference sequences, but propose a new *opposite*
     module in at least one corresponding slot. Require a verifiable inverse
     semantic direction/recipe relation (for example long-vs-short edge
     preference, usage decay-vs-boost, or compress-vs-amplify). No change is
     invalid.
   - `p3`: use **only the base sequence** as its reference; revise at least
     one selected module's design sentence and implementation, without using
     the old parent sequence. Keep the public slot contract.
   - `p4`: use **only the base sequence**; keep every slot's module ID and
     description/code fixed, change at least one bounded parameter value,
     and verify that the compiled source changes. No new module is registered.
   For p1/p2/p3, the new module must become a registered, searchable member
   of its slot and be inserted into the child's K=3 set. Reject invalid or
   duplicate proposals, retry within a limit, and log failures explicitly.
4. Collect all module proposals, validate and register them in deterministic
   parent/operator order, then compile/evaluate offspring in parallel.
   Workers must not mutate the registry while velocity/cut computation or
   other workers are reading it. The generation has N*len(operators) planned
   child records; failed generation/evaluation is represented explicitly.
5. Apply EoH-style elitism: rank previous N parents together with the valid
   children, keep the best N distinct positions. A surviving child inherits
   its parent's updated velocity and historical pbest, updates pbest if its
   own objective is better, and carries a new stable particle ID plus parent
   ID. Unselected parents/children are not silently substituted. Recompute
   gbest, preserve ties by position identity, and report both generated-child
   count and survival count per operator.

## Logs, tests, and reproducibility

- Every attempted sample record includes `generation`, `sample_id`, stable
  `particle_id`, `parent_id`, `operator`, `base_sequence`, final sequence,
  module IDs/descriptions/recipes/code hashes, alpha and raw/constructed cut,
  objective, evaluation details, validation/failure reason, and prompt/response
  IDs. Checkpoints save the evolving module library, position/set/velocity,
  pbest/gbest, RNG state, operator list, and next IDs. Resume must reproduce
  the next generation without regenerating prior modules or prompts.
- Use fake deterministic LLMs and cheap evaluators in unit tests. Verify ten
  valid generated modules in each slot, 20->10 initialization, exactly
  N*len(operators) attempted offspring for `[p0]`, `[p1,p2]`, and
  `[p1,p2,p3,p4]`, operator provenance, relation constraints, p4 parameter-only
  behavior, K=3 and position-membership invariants, description-on/off
  prompt difference, ties, checkpoint/resume, and invalid-LLM handling.
  Add a small TSP GLS contract smoke test for a generated module and compare
  scoring with the existing evaluator. Do **not** launch the 64-instance,
  20-generation online experiment during tests or expose the configured key.
