# PREREGISTERED.md: decision rules and their outcomes

Several decisions in the later stages of this work were made against rules
fixed before the deciding number existed. A threshold chosen after the data had
been seen would describe the data rather than test it. Until this register was
written, those rules existed only in discussion, while the project's internal
report recorded the outcomes: a reader of that report could see that a rule
fired, but could not check what it said, or whether it was edited on the way.

This file is the register. Each rule is stated in the form in which it was
fixed, with its date, followed by what happened, including the case of a rule
that fired correctly and still gave the wrong answer (R2).

**Scope.** This is a working register for the later stages of this study,
written from the record after the rules were fixed. It is not a claim of formal
preregistration with a third-party timestamp, and one rule (R2) survives only
in paraphrase, which is noted at the rule.

| id | rule | fixed | fired | outcome |
|---|---|---|---|---|
| R1 | binary-vs-ternary CI band, ±1.0 pp | 2026-08-04 | 2026-08-05 | fired; **equivalence branch did NOT trigger** |
| R2 | SSL codebook-collapse alarm | 2026-08-03 | 2026-08-04 | fired; **true positive, wrong diagnosis**, amended |
| R3 | SSL full-run arm selection | 2026-08-03 | 2026-08-04 | fired twice; **the 100k firing reversed the 20k one** |
| R4 | 20k pilots are sensitivity screens | 2026-08-04 | standing | scoping rule, not a test |
| R5 | loader-migration acceptance | 2026-08-06 | 2026-08-06 | **PASSED 9/9** once GPU family is matched |
| R6 | probe-trajectory reading | 2026-08-06 | 2026-08-06 | applied; **no crossover, neither arm converged, taps disagree** |
| R7 | FSQ-vs-RVQ control comparisons under a matched training objective | 2026-11-16 (addendum same day) | 2026-11-18 | applied; **C1: no detectable difference; C2: 8-layer RVQ higher by 0.67 pp** |

---

## R1: the binary-vs-ternary CI band (±1.0 pp)

**Fixed 2026-08-04**, before either binary arm reached n = 5, and therefore
before any of the six matched-rate intervals existed.

**Statement.** Compare each binary arm against the measured ternary brackets
`[3]^6` (9.51 bits) and `[3]^7` (11.09 bits), which straddle the binary arms'
10.00 bits, using a paired t-test on matched seeds. Read the 95% CI on the mean
difference, and take exactly one of three branches:

- **excess**: the interval lies entirely above 0. Binary buys accuracy at
  matched rate. This is the claim the earlier chord construction made.
- **deficit**: the interval lies entirely below 0.
- **tight equivalence**: the interval contains 0 **and lies entirely within
  ±1.0 pp**. Only this branch licenses the word *equivalent*.

An interval that contains 0 but runs outside ±1.0 pp is a **null**, not an
equivalence, and must be reported as "no detectable difference, with an effect
of up to X pp not excluded".

**Why ±1.0 pp.** It is the width below which a difference does not matter for
the argument of the paper, chosen against the seed-to-seed SDs of the
configurations involved (0.4 to 1.4 pp) rather than against any observed
difference.

**What happened** (2026-08-05). Five of six comparisons took the **deficit**
branch: the earlier claimed excess has the opposite sign, not only a different
magnitude. The sixth, `[2]^10 sym − [3]^6`, gave **+0.34 pp, 95% CI [−0.80,
+1.49], p = 0.45**. The interval contains 0, so it is not a deficit; its upper
bound is +1.49 pp, so it lies **outside the band** and the tight-equivalence
branch does **not** fire.

**Outcome.** The tight-equivalence branch did not fire. A difference of
+0.34 pp with p = 0.45 can be read as equivalence; the band rules that reading
out, because the data do not support it. Reported as a null.

---

## R2: the SSL codebook-collapse alarm

**Fixed 2026-08-03**, when the 100k alpha = 0 run was queued and before any of
its checkpoints existed. It was set up because the failure mode expected at
alpha = 0 is a *slow* collapse, which a 20k pilot cannot show. This is also why
`train_ssl_w2v2.py` writes per-layer perplexity into every periodic checkpoint.

**Statement, as it survives in the record.** Watch per-layer perplexity across
the periodic checkpoints of the alpha = 0 run. A sustained fall on all four RVQ
layers, trending toward the supervised RVQ-4L's layer-0 value of ~13.8, is the
collapse signature; on that signature, escalate: treat the run as failing and
do not let the alpha = 0 arm carry a manuscript claim without a second arm.

**Note on the record.** The alarm was a monitoring rule agreed in discussion, not
code. Its qualitative form and its escalation clause are recorded in the
project's internal report; the exact numeric trigger is **not** recoverable
from the project's repository. R5 and R6 below are written in a form that does
not have this problem.

**What happened** (2026-08-04). The alarm fired on all four layers at the 30k
boundary: SOFT perplexity at layer 0 went from 119.5 to 88.6 and then to 61.7.
Under the rule as written that is a correct trigger.

**Diagnosis.** The alarm's premise, that falling perplexity means the codebook
is dying, was contradicted by the rest of the instrumentation at the same
checkpoints:

| boundary | SOFT (per-step) | HARD (eval-set) | used codes / 128 |
|---|---|---|---|
| 10000 | 119.5 | 12.1 | 23 |
| 20000 | 88.6 | 14.2 | 27 |
| 30000 | **61.7** | **17.5** | **34** |

Collapse requires all three to fall together. Two of the three were **rising**:
the encoder was covering more distinct codes over the eval set while its
per-frame assignments grew more confident, the pattern that the amendment below
labels *sharpening*. The escalation clause rested on an assumption that had not
been checked, that low SOFT perplexity is pathological, and the supervised
RVQ-4L contradicts it: it sits at 13.8 at L0 and is the best-performing RVQ
configuration in the panel.

**Amendment, adopted 2026-08-04 and binding from then on.**

1. The trigger is **three-signal, not one**: SOFT perplexity, HARD (eval-set)
   perplexity, and used-code count must **all** fall together. Any signal moving
   the other way disqualifies the collapse reading and the observation is
   labelled *sharpening* instead.
2. **SOFT and HARD are never compared to each other**. The per-step
   `perp` field lives at ~114 to 128; the checkpoint's `per-layer perplexity` line
   is a different quantity living at ~8 to 15. The 13.8 supervised reference is a
   SOFT value and may only be compared against SOFT.
3. **The probe adjudicates, not the perplexity trace.** Until a probe number
   exists, neither "collapse" nor "sharpening" may appear in manuscript-facing
   text for that run.

Under the amended rule the alarm does **not** fire on the 100k alpha = 0 run:
HARD and used-codes rise. The refuted reading is kept in the register because it
shows a failure mode of such alarms: a correct trigger on a premise that had not
been checked.

---

## R3: SSL full-run arm selection

**Fixed 2026-08-03**, before the 20k alpha ablation returned.

**Statement.** The paper's RVQ arm is the one that wins the frozen linear probe
on **both** criteria: final-layer (`layer_6|linear`) accuracy **and** best-tap
accuracy. If the two criteria disagree, the tie-break is best-tap, on the
grounds that the question is what the encoder *represents*, not what survives to
its last layer. The comparison is decided **at the paper's training budget**,
i.e. at 100k steps.

**First firing, 2026-08-03, at 20k.** alpha = 0 won both criteria
(26.36% / 28.21% against 25.81% / 27.92%), so the tie-break was not needed.

**Second firing, 2026-08-04, at 100k.** Both arms
were run to 100k and probed at `best.pt` (step 86000 for both):

| arm | final-layer | best tap (L1 for both) |
|---|---|---|
| alpha = 0 | 29.91% | 38.64% |
| **alpha = 0.01** | **31.76%** | **40.16%** |

**alpha = 0.01 wins both criteria, reversing the 20k selection.** Under the
rule's "at the paper's training budget" clause the 20k firing was provisional
(R4 gives the reason), so the second firing is the decision. Re-probing at
`final.pt` did not change the ordering, so the checkpoint choice does not affect
the selection.

---

## R4: the 20k pilots are sensitivity screens, not budget-valid measurements

**Adopted 2026-08-04.** A scoping rule rather than a
test: it has no branches and cannot fire, but it constrains what every 20k number
in the project's internal report is allowed to say.

**Statement.** Both the 20k pilots and the 100k runs use OneCycle over their own
horizon, so at step 20k the pilot is at the **end** of its schedule with the LR
annealed to near zero while the 100k run is near **peak** LR with 80k steps
left. Matching values at matched step counts is therefore a coincidence of
schedule position, not agreement. Consequently:

- a 20k number may be used to **rank** configurations under a common cheap
  budget, which is what the pilots were for, and why the RVQ × alpha
  interaction survives, being a comparison *among* pilots;
- a 20k number may **not** be quoted as the behaviour of that configuration at
  the paper's budget without a 100k check;
- this applies symmetrically. The alpha = 0.01 arm's "pinned at 128.0/128" is a
  20k reading under exactly the same caveat, which is why it was measured at
  100k rather than kept as a safe fallback, and it is the arm that then won.

**Consequence already realised.** The internal report's "removing L_d does not
collapse the SSL codebook" was rewritten from FINAL to AMENDED (scoped), and R3's
second firing followed directly from applying this rule instead of trusting the
20k firing.

---

## R5: acceptance rule for the loader migration

**Fixed 2026-08-06, before the check job was submitted**, that is, before any
new-path number existed. Recorded here and enforced in code by a comparison
script, so that, unlike R2, this rule does not survive only in paraphrase.

**Statement.** Migrating every evaluator onto `load_model_from_checkpoint`
changes which *class* is constructed: the trainer's, rather than an eval-side
clone. The clones are not literally the same code (`train_rvq_supervised`'s
ResidualVQ recomputes its residual incrementally where the clone recomputed it
from `z`, which can flip an `argmin` at a near-tie), so the migration is a
behaviour change until proven otherwise.

The claim to be tested is therefore narrow: **the same checkpoint, on the same
manifest, produces the same numbers**. Since evaluation here is deterministic,
the only honest criterion is **exact equality** (not "close", not "within
noise") on:

- `overall_accuracy`;
- every per-BPC field, all 8 classes;
- every per-phoneme field, all 39 phonemes;
- the scalar `rate_analysis` fields, where both sides have them.

Provenance fields (`eval_job_id`, `commit_hash`) and the new `load_provenance`
block are excluded, being expected to differ.

**Branches.** All nine cases identical means the migration is a refactor and the
recorded numbers stand unchanged. **Any** differing field in **any** case means the
migration is a numerical change; the differing configuration is named, the
affected numbers in the internal report are marked as superseded, and the change
is defended on its merits or reverted. A non-empty diff is never absorbed
silently.

**Coverage.** Nine cases for the eight `--model_type` values. The ninth is
`fsq_bin10d_sym`: the same `--model_type fsq`, but the symmetric-bound
checkpoint whose silent mis-load (13.04% against a true 73.57%)
is the reason the loader exists, so it is tested directly rather than by
inference from the asymmetric arm.

**What happened, 2026-08-06, the check job, `DRY_RUN=0`, on an A100 node.
Seven of nine identical to the last recorded digit; two differ:**

| case | old | new | delta | frames of 800,813 |
|---|---|---|---|---|
| `fsq_bin10d_sym` | 73.57285658449601 | 73.57298145759373 | +0.000125 pp | 1 |
| `fsq_4d_8555_bpc8` | 63.722367144389516 | 63.72199252509637 | −0.00037 pp | 3 |

The rule says a non-empty diff is never absorbed silently, so the difference
was traced. **The two differing cases are exactly the two whose recorded JSON
was produced on a different GPU architecture**, with no exceptions in either
direction:

| case | original node | GPU | verdict |
|---|---|---|---|
| `fsq_bin10d_sym` | L40S node 1 | **L40S** | differs |
| `fsq_4d_8555_bpc8` | L40S node 2 | **L40S** | differs |
| the other seven | four A100 nodes | **A100** | identical |

`evaluate_fsq.py` never set `cudnn.deterministic` (its `--seed` is documented
as provenance only), so kernel selection is free to differ across GPU families.
Both affected configs use even FSQ levels, whose bound rounds after an extra
shift/scale/offset; that is a plausible mechanism but is **perfectly confounded
with GPU family across these nine cases** and is not claimed as established.

**Confirmatory control: a second job, `NVIDIA L40S`, L40S node 2** (the
same node that produced one of the two recorded JSONs). The two differing cases
re-run against the same recorded JSONs:

```
fsq_bin10d_sym    identical  (73.5729% over 39 phonemes)
fsq_4d_8555_bpc8  identical  (63.7224% over 39 phonemes)
2/2 cases reproduce exactly.
```

**Status: PASSED, 9/9.** Every case reproduces exactly when the GPU family
matches the one its recorded JSON came from. The migration is a refactor, as
claimed; the residual is a property of the hardware. The first branch applies
and the recorded numbers stand unchanged.

The rule has no "close enough" branch, so a three-frame difference in 800,813
was followed up by a controlled experiment, which also produced a second,
unrelated result about the evaluation protocol that a tolerance would have
hidden.

---

## R6: how the probe-vs-step trajectory will be read

**Fixed 2026-08-06, before the trajectory probe job returned.** The trajectory
is the one piece of evidence in this comparison that has never been measured,
so the reading is fixed first; the trajectory collection script computes
exactly these quantities and nothing else.

**What is measured.** Ten checkpoints per arm (`step_10000` ... `step_90000`, then
`final.pt` at 100k) for both 100k runs, FSQ [3]^16 and RVQ-4L ×128
(alpha = 0.01), probed at `layer_6|linear` and `layer_1|linear`,
10000 probe steps each, the same budget as every published probe number.

**Reading 1: does the lead change?** Compute the FSQ-minus-RVQ gap in pp at
each shared step, at each tap. If the sign changes at any step, the arms swap
order and the manuscript may not say either arm "leads": it must name the
crossover step. If the sign is constant, the arm ahead at 100k led throughout,
and that is a stronger sentence than the endpoint alone supports.

**Reading 2: has either arm converged?** An arm whose peak accuracy is at its
last point is still rising at 100k. If either arm is still rising, the 100k
comparison is a snapshot of two curves that have not converged and must be
labelled as one; only if both have peaked before 100k may it be described as a
comparison of two converged models.

**Reading 3: the taps may disagree, and that is a result.** Earlier probes
found that RVQ does not recover at mid-stack and that L1 is the best tap for both arms. If the
final-layer and L1 trajectories give different answers to Reading 1, the
disagreement is reported rather than resolved in favour of whichever tap the
selection rule (R3) happens to use.

**Not pre-registered, deliberately.** No threshold is fixed on the *size* of the
gap. There is one seed, so a gap has no interval; the trajectory is read for
shape and ordering, which one seed can support, and not for magnitude, which it
cannot.

**What happened: the trajectory probe job, 20/20 probes, 0 failed, 2:37:29.**

- **Reading 1: no sign change at either tap.** FSQ leads `layer_1` at 10 of 10
  steps; RVQ leads `layer_6` at 10 of 10. There is no crossover step to name, so
  the rule allows the stronger sentence: at each tap the arm ahead at 100k **led
  throughout**.
- **Reading 2: neither arm has converged at `layer_1`**: both peak at their last
  point. The 100k comparison is a snapshot of two curves still climbing and is
  labelled as one. At `layer_6` both are flat after ~80k; the strict argmax says
  "past peak" but by **0.004 pp** and **0.031 pp**, which is a tie at the
  reporting precision. A decline would be a magnitude claim, which this rule
  does not allow, so it is reported as flat, and the collection script now
  prints the shortfall beside the boolean.
- **Reading 3: the taps disagree completely, and this is reported, not resolved.**
  Opposite orderings at the two taps, stable across all of pretraining. The
  arms gain comparably at L1 (+10.02 / +8.93 pp) and differ almost entirely in
  whether that gain survives to L6 (+1.17 / +7.52 pp), so the final-layer gap
  arises during pretraining rather than being an initial condition.

**Effect of the "not pre-registered, deliberately" clause.** The endpoint gaps of
this job are +2.32 pp and −7.76 pp. Because no threshold on gap *size* was fixed
in advance, neither appears as an effect size: one seed supports signs,
orderings and monotonicity, and nothing more.

---

## Rules that are conventions, not decisions

Recorded for completeness. These constrain the work but have no branches, so
nothing can "fire":

- **Canonical condition.** Every paper-facing number is dev-clean, eval mode,
  30k steps, cuDNN-deterministic, with the seed stated. No manuscript claim from
  single-seed numbers without labelling n.
- **Frozen trainers.** `scripts/train_fsq.py` and `scripts/train_rvq_bpc.py`
  are never edited; odd-L FSQ must stay bit-identical to the published path.
  Changes go through a launcher (`train_fsq_sym.py`).
- **Multiplicity.** The six paired tests of the multi-seed analysis (identity −
  FSQ [3]^16, [3]^8 − [8,8,5,5,5], [3]^64 − FSQ [3]^16, [3]^64 − [3]^32,
  identity − [3]^64, [2]^10 sym − [2]^10 asym) are one family and carry Holm
  adjustment; decisions are read off the adjusted p.
- **A job ID is not a record of what was launched.** Every submission
  report states `DRY_RUN` / `ALPHA` / `SEED` beside the ID; the authoritative
  record is the job's own startup echo. Enforced at generator level by the
  `dry_` job-name prefix.
- **Seed-42-only displays.** Confusion matrices and t-SNE remain seed 42
  (averaging them across seeds is not a defined operation), and their captions
  must say so. The panel *statistics* around them are at n = 5.

---

## R7: Control comparisons for the FSQ versus RVQ claim under a matched training objective.

**Fixed 2026-11-16**, before any control run was submitted.

**Background.** The arm reported as "RVQ 4L x128 supervised"
(`ART/outputs/multiseed/rvq_4l`) was trained with `train_rvq_bpc.py`: BPC-8
cross-entropy x1 + phoneme cross-entropy x0.5 + VQ. FSQ [3]^16
(`ART/outputs/multiseed/fsq_16d`) was trained with phoneme cross-entropy only.
The reported difference between them (+2.10 pp) therefore differs in objective
as well as in quantiser. The RVQ layer and codebook sweeps used the same BPC-8
objective. Source: the project's internal records.

**New arms.** Five seeds each (42, 123, 456, 789, 2024), 30k steps,
cuDNN-deterministic training, standalone dev-clean evaluation of `best.pt`.
Trainer and evaluator files unchanged.

- **A.** `rvq_4l_supervised`: `train_rvq_supervised.py`, 4 layers x 128 codes
  (phoneme CE + VQ).
- **B.** `rvq_8l_supervised`: the same with 8 layers.
- **C.** `fsq_16d_bpc8`: `train_fsq.py`, levels [3]^16, BPC-8 supervision, fine
  weight 0.5.
- **D.** `fsq_16d_rerun`: FSQ [3]^16, phoneme CE only, same arguments as
  `multiseed/fsq_16d`.

**Comparisons.** Paired by seed, two-sided paired t-test, mean difference with
95% CI and raw p. None is added to the Holm family of six.

- **C1 (primary):** FSQ [3]^16 minus A on dev-clean, repeated on dev-other,
  test-clean and test-other. It replaces the +2.10 pp comparison whatever its
  sign and size.
- **C2:** FSQ [3]^16 minus B. It replaces the +0.84 pp comparison against
  RVQ-L8.
- **C3:** C minus the existing `multiseed/rvq_4l` arm (both under the BPC-8
  objective).
- **C4 (descriptive, effect of the objective within a quantiser):** A minus
  `multiseed/rvq_4l`; B minus `multiseed_phase2/rvq_L8`; existing FSQ [3]^16
  minus C.
- **C5 (replication):** D minus the existing `multiseed/fsq_16d` arm. If the 95%
  CI of C5 excludes zero, C1 and C2 are reported with D as the FSQ arm and the
  discrepancy is reported. Otherwise the existing FSQ [3]^16 values are used and
  the D-based values of C1 and C2 are given as a sensitivity analysis.

**Reading rule for C1, C2 and C3.** If the 95% CI excludes zero, a difference is
reported with its sign. If the 95% CI lies within +/-1.0 pp (the band of R1),
the arms are reported as equivalent within 1 pp. If both hold, the difference is
reported and stated to be within the band. Otherwise "no detectable difference"
is reported with the CI.

**Run handling.** All five seeds of every arm are reported. A job that fails for
an infrastructure reason is resubmitted with the same seed and arguments. No
seed is added, replaced or dropped on the basis of its accuracy.

**Rate analysis.** Entropies, unique-code counts and utilisation are recomputed
with (S) a mask keeping the dev-clean frames scored for accuracy and (L) a mask
keeping all real frames. The S values replace those in the manuscript; the L
values are reported beside them. The script must reproduce the stored unmasked
values before masked values are accepted.

**What happened.** See the entry of 2026-11-18 after the addendum.

**Addendum, 2026-11-16, before any control run was submitted.**

**Cluster migration.** Between the existing runs and these, the cluster was
migrated (nodes renamed, new operating system and
module tree). The control launchers differ from their sources as follows; no
training or evaluation argument changes:

- the node exclusion uses the new names of the same nodes;
- the cuda and cudnn module lines are removed because those modules no longer
  exist; the torch 2.7.1+cu118 build supplies CUDA 11.8 and cuDNN 9.1.0;
- time limits are 2 h 30 (training), 1 h (held-out evaluation) and 1 h 30
  (masked rate analysis);
- the masked rate job may run on A100, H200 or L40S nodes; its self-check
  tolerances are 1e-3 bits for entropy fields, 0.1% or 10 codes (whichever is
  larger) for the unique-code count, and 40 frames for accuracy;
- jobs may also be submitted to a pre-emptible partition; a pre-empted job is
  rerun from step 0 with the same seed and arguments.

Arm D (comparison C5) is the check that the migrated environment reproduces the
existing FSQ [3]^16 values.

**Availability rule.** The paper is due on 2026-11-24. Results available
by 2026-11-21 18:00 UTC are used.

- C1 is reported only if all five seeds of arm A have completed by then.
  Otherwise the manuscript is corrected from existing data only, and the
  completed runs are reported descriptively.
- If arm D is incomplete, C1 is reported against the existing FSQ [3]^16 arm and
  the completed D seeds are reported descriptively.
- An incomplete arm B or C is reported as incomplete and supports no claim.

**What happened: 2026-11-18** (every run from one commit).

- **Runs.** All four arms completed at five seeds, each with its standalone dev-clean evaluation, and arm A was evaluated on dev-other, test-clean and test-other. Every result was on disk by 2026-11-18 08:14:39 UTC, before the cut-off. No seed was added, replaced or dropped. Dev-clean accuracy, mean and sample SD: A 79.60 (0.55), B 80.47 (0.40), C 78.04 (0.43), D 79.63 (0.59), existing FSQ [3]^16 79.80 (0.46).
- **C5 (replication).** D minus the existing FSQ [3]^16 arm: -0.17 pp, 95% CI [-0.39, +0.05], p = 0.098. The CI includes zero, so the existing FSQ [3]^16 values are used and the D-based values of C1 and C2 are the sensitivity analysis.
- **C1 (primary).** FSQ [3]^16 minus A on dev-clean: +0.20 pp, 95% CI [-0.87, +1.27], p = 0.63. dev-other: +0.21 pp [-0.87, +1.29]. test-clean: +0.23 pp [-0.81, +1.27]. test-other: +0.19 pp [-0.94, +1.32]. Reading on every split: no detectable difference. No CI lies within +/-1.0 pp. With D on dev-clean: +0.03 pp [-1.18, +1.24]. C1 replaces the +2.10 pp comparison.
- **C2.** FSQ [3]^16 minus B: -0.67 pp, 95% CI [-1.25, -0.09], p = 0.032. Reading: a difference, with the 8-layer RVQ higher; the CI does not lie within +/-1.0 pp. With D: -0.84 pp [-1.60, -0.09]. C2 replaces the +0.84 pp comparison.
- **C3.** C minus `multiseed/rvq_4l`, both under the BPC-8 objective: +0.34 pp, 95% CI [-0.54, +1.22], p = 0.35. Reading: no detectable difference.
- **C4 (descriptive).** A minus `multiseed/rvq_4l`: +1.90 pp [+0.55, +3.26]. B minus `multiseed_phase2/rvq_L8`: +1.52 pp [+0.97, +2.06]. Existing FSQ [3]^16 minus C: +1.76 pp [+0.96, +2.57]. Within each quantiser, the arm trained on phoneme labels alone is higher than the arm trained with the BPC-8 objective.
- **Rate analysis.** The masked script reproduced the stored unmasked values within the addendum's tolerances for all 44 quantised entries. The S and L values are in section E of the project's internal results file.
- **Deviations.**
  1. Partitions and nodes. The first submission (2026-11-16) went to the standard and the pre-emptible GPU partitions. The 15 jobs of arms A, B and D were pre-empted in the pre-emptible partition, and the five arm-C jobs failed with a CUDA out-of-memory error on a 4.75 GiB MIG slice of one node. All were resubmitted with the same seeds and arguments, with that node added to the node exclusion. From 2026-11-17 12:33 UTC the training jobs were submitted to the standard GPU partition only.
  2. Masked self-check. On an NVIDIA A100 80GB the record of `ms_fsq16d_s456` failed one entropy check by 1.1e-5 bits against the 1e-3 tolerance; the stored analysis had been computed on an NVIDIA L40S. Under a decision recorded on 2026-11-17 13:38 UTC, before any masked value of that checkpoint existed, the record was moved unchanged to `superseded/` and recomputed once on an L40S, with that result final. The L40S record passed all 11 checks. The masked folder therefore holds 40 records computed on an L40S and 7 computed on an A100 80GB.
  3. Values seen early. On 2026-11-18 at 10:47:00 UTC, after every job had ended and before the completeness test finished (10:48:39), a log check printed the per-class counts of two runs (A seed 456, C seed 456). No scheduler action was taken afterwards and no step depended on those values.
  4. GPU types. Arm A ran on A100-SXM4-40GB cards only. Arms B, C and D each mix A100-SXM4-40GB and L40S cards across seeds.
  5. Fingerprints. The C and D checkpoints carry no configuration fingerprint, because `train_fsq.py` writes none; they loaded with 0 missing and 0 unexpected keys.
  6. Checkpoint selection. In all 20 runs `best.pt` is the step-28,000 checkpoint, the last validation.

**Note added 2026-11-18, after the entry above.**

- **Commit.** "Every run from one commit" in the entry refers to the control runs. The existing arms used in C1 to C5 come from three earlier commits: one for `multiseed/fsq_16d` and `multiseed/rvq_4l`, one for `multiseed_phase2/rvq_L8`, and one for the held-out evaluation of FSQ [3]^16.
- **Deviation 1, further detail.** The first full masked job was also pre-empted in the pre-emptible partition, after writing 8 records. The move to the standard GPU partition only was made by cancelling and resubmitting 18 queued jobs. Arm A seeds 42 and 123 had been submitted to both partitions and ran in the standard GPU partition.
- **Rate analysis, controls.** The three seed-42 control records (arms A, B and C) had no stored analysis and were compared with their in-job evaluations. A and C are identical to theirs. B differs by up to 1.645e-3 bits in a joint-entropy field and by one unique code; its in-job evaluation ran on an L40S and the control record on an A100-SXM4-40GB.
- **Displays.** The t-SNE and confusion displays were not regenerated: the arm A seed-42 analysis job had not started 3 hours after submission and was left queued.

**Note added 2026-11-19, after the note of 2026-11-18.**

- **Rate analysis, addition.** Masked records for FSQ [3]^6 and [3]^7 (seed 42) were computed by a separate job on NVIDIA H200, with the in-job evaluation as reference. Outcome: [3]^6 OK, [3]^7 OK.
- **Displays.** The arm A seed-42 analysis for the t-SNE and confusion displays was computed by that queued job on NVIDIA H200, which started 17 hours 15 minutes after submission. The three displays were regenerated from it.
