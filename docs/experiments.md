# Experiment log

## What this log is for

Two questions, and only the second one is a contribution:

1. Does MoCo v2 transfer to CT colonography at all?
2. **Which of its assumptions break on CT, and what replaces them?**

Phase 0 answered (1) negatively: naive MoCo v2 on CT loses to off-the-shelf
ImageNet weights. That is a result but not a finding. The finding has to be an
*attribution* — this specific augmentation assumption is wrong, and replacing it
with this specific alternative recovers this much. A ladder of runs that each
change one thing is the only way to make that claim; a bundle of changes shipped
together can only say "something in here helped".

## Protocol

1. **Pre-register before submitting.** Write the entry's *Hypothesis* and
   *Prediction* rows before the job runs, not after it lands. A prediction
   written afterwards is a story, and the whole point of the ladder is to be able
   to say we called it.
2. **One change per run**, stated as a diff against the named parent run.
   A run that changes two things gets a letter suffix and is treated as a
   bundle, not a rung.
3. **Numbers are never typed by hand.** `eval_repr.py` writes JSON;
   `scripts/eval/log_experiment.py` flattens it into
   [`experiment_results.csv`](experiment_results.csv) and prints the markdown
   row. Paste that row. Transcription is where comparisons rot.
4. **Byte-identical input.** Every encoder is scored on the same crop bank
   (`/scratch/$USER/moco/eval/bank.npy`, 33,648 crops from 2,103 series since the
   2026-09-16 rebuild). If the
   bank is ever rebuilt, every prior run is re-scored against the new one or the
   comparison is void.
5. **Record failures and kills in full.** A run that hit a kill gate at epoch 20
   is evidence about the augmentation, which is exactly what we are measuring.
   Deleting it loses the finding.

### Kill gates

Stop a pretraining run early if the gate fires. It means the pretext task is too
easy, which is the failure Phase 0 diagnosed, and it does not need the full run
to detect:

- `Acc@1` still above **95% at epoch 50**. Healthy contrastive training sits
  around 60-85%. **Amended 2026-09-17 after E0**, which sat at 0.4% at epoch 20
  (chance is 0.39%) and still reached 99.9%: with per-worker seeding the curve
  now takes off between epochs 20 and 49, so the epoch-20 reading says nothing.
  E0's numbers by epoch: 0.4 (20), 94.1 (49), 99.8 (99), 99.9 (199).
  It worked as amended on E1: 37.8% at epoch 49.
- ~~Augmented-view cosine (`alignment_cos`) above **0.99**, scored on the
  epoch-20 checkpoint.~~ **Withdrawn 2026-09-21 after E1**, which scored higher
  on it than E0 (0.9936 vs 0.9917 at epoch 20; 0.9958 vs 0.9971 at epoch 200)
  while beating E0 on every downstream measure. A high cosine does not signal
  collapse here. `alignment_cos` stays in the ledger as a logged number only.

### The bar

**ImageNet cross-position retrieval top-1 = 0.494** (P0r, the rebuilt bank; was
0.493 on the P0 bank). Label-free, ACRIN-only, n=613 queries against a gallery
of 671, standard error about 0.02, so it resolves a 5-point change. Chance is
0.0016. **E1 cleared it (0.873, 2026-09-21).**

From E2 on, each rung is judged against its parent, not against ImageNet. A gap
counts only if it exceeds both the sampling threshold (~0.06) and the retrain
noise E1b measures. Near E1's level the sampling SE shrinks to about 0.013, but
headroom does too: E1 leaves 0.127 of top-1 and 0.018 of top-5, so later rungs
may be hard to separate on this metric. See the open questions under E1's result.

Secondary, and not difficulty-matched to each other: `any_series` top-1 (same
task over the full ACRIN series pool, where alternate reconstructions of one
acquisition make it winnable by near-duplicate matching) and RankMe. **RankMe is
a floor detector only** (it catches collapse): it fell on E0 and rose on E1 while
both improved, so its direction is not read.

`knn_polyp` is reported but **cannot discriminate encoders** — the validation
split was 45 patients with 3 in the large class (51 with 5 after the 2026-09-16
re-split, which does not change the conclusion), so the metric's resolution is
about 0.1 balanced accuracy against an observed encoder spread of under 0.03. It is
logged for completeness until Phase 3 replaces it.

### How runs are measured, and how far to trust it

**The procedure.** Training is finished and the encoder is frozen. It turns each
of the 33,648 crops in the bank into a vector, and the 16 vectors from one
series are averaged into one vector for that scan. Every model sees exactly the
same crops.

**The main number.** Take a patient's supine scan and ask which of the 671
prone scans is most similar. Correct means it picks the same patient's prone
scan. `cross_position_top1` is the fraction of 613 supine scans that get it
right. Between the two scans the patient rolled over, so gas, fluid and
collapsed bowel have all moved and the pixels do not line up. What stays the
same is the patient's anatomy, so a model can only score well by encoding
anatomy. Chance is 0.16%, ImageNet gets 49%, the old MoCo 32%, E1 87%.

**Where the evaluation scans come from, and how that differs from a
train/val/test split.** The crop bank (`build_crop_bank.py`) is not a sample.
It holds **every cached volume**: all 1,744 ACRIN and 359 Pediatric series,
2,103 in total, 16 crops each. Crops are placed deterministically: at depth
fractions (i + 0.5) / 16, and centred in-plane. Nothing is drawn at random, so
a rebuild reproduces the bank exactly. Cross-position retrieval then uses the
ACRIN part only. The 613 queries are every supine series whose patient also has
a prone series, and the gallery is all 671 prone series. Nobody chose these
scans. They are every supine/prone pair in the collection.

So the retrieval metric is **not a held-out test set** in the train/val/test
sense. MoCo pretrains on every ACRIN volume, and those same volumes are scored.
That is deliberate and standard for self-supervised evaluation: MoCo never sees
a label, and it never sees a supine and a prone scan paired as the same
patient. It only pairs crops from within one series. The retrieval task asks
something training never asked. The costs are known, and each one works
against us rather than for us:

- *It favours MoCo over the controls*, which never saw these scans. P0 lost
  anyway. E1's win over ImageNet carries this caveat and should be reported
  with it.
- *It stops being valid the moment training sees cross-position pairs.* E5
  (supine/prone positives) would train directly on the 613 query/answer pairs,
  so E5 cannot be scored on this metric as it stands. That has to be settled
  before E5 runs, e.g. by holding a patient set out of E5's pretraining and
  scoring retrieval only on those patients.
- *It shares patients with the frozen polyp test split.* `labels_test.csv` (51
  patients) is frozen for labels only. Their scans are in every pretraining run
  so far. Phase 3's step 5 already excludes each test fold from pretraining,
  and that is where this is resolved. Until then, "no encoder has seen its test
  scans" is true of labels, not of pixels.

The labelled splits (`split_data.py`, seed 42, patient-disjoint, stratified by
polyp class: 243 train / 51 val / 51 test patients) are the conventional
train/val/test split. They serve only the label-based probes: `knn_polyp`
fits on train and scores on val, and nothing reads test until Phase 3. The
confounder and collection probes use their own random patient-disjoint halves
(eval seed 0), and `knn_zpos` splits a random 8,000-crop sample in two.

**Why it can be trusted:**

- *Same input for everyone.* One fixed crop bank, so a difference comes from
  the encoder, not from which crops it happened to get.
- *Two controls scored identically.* Random init shows what the architecture
  gives for free; ImageNet is the bar a CT-specific pretraining has to clear.
- *Known sampling noise.* With 613 queries the standard error is about 0.02, so
  two runs have to differ by about **0.06** before the gap is more than noise.
- *Predictions written first.* Each rung states its expected result before it
  runs, so a result cannot be explained after the fact.
- *Cross-checks that disagree for known reasons.* The epoch-50 accuracy gate
  catches a too-easy pretext task, RankMe catches collapse, and the confounder
  probe catches an encoder sorting scans by bowel prep instead of anatomy.

**What it does not tell us:**

- ~~*Retraining noise is unmeasured.*~~ *Measured 2026-09-25 (E1b):* E1
  retrained with a new seed scored 0.874 against 0.873, so at epoch 200
  retrain noise on this metric is well under sampling noise. It is larger on
  `any_series` (0.04) and `knn_polyp` (0.02), and very large at epoch 20
  (0.50 vs 0.16). See E1b's result.
- *It is a proxy for anatomy, not for polyps.* A model can recognise a patient
  without seeing a polyp. The ladder uses this number to *choose a recipe*
  because it is label-free and has 613 queries. Whether that recipe helps find
  polyps is Phase 3's job, answered with labels.

---

## P0 — Phase 0 baseline (2026-09-03, job 62516893)

Frozen encoders, no training. Establishes the "before" measurement.

| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | align cos | zpos MAE | polyp bal-acc |
|---|---|---|---|---|---|---|---|
| P0:random | 0.023 | 0.05x | 0.052 | 16.7 | 0.9991 | 0.1084 | 0.333 |
| P0:imagenet | 0.493 | 1.00x | 0.292 | 666.7 | 0.9602 | 0.1093 | 0.329 |
| P0:moco checkpoint_0199 | 0.318 | 0.65x | 0.183 | 330.2 | 0.9973 | 0.1185 | 0.319 |

**Findings.**

- MoCo loses to ImageNet on every anatomy metric, landing at a consistent
  63-65% of it. Unpaired z = 6.3 on cross-position, so not a reseed effect.
  MoCo pretrained on 100% of this bank and both controls saw none of it, so the
  comparison is transductive in MoCo's favour and it lost anyway.
- It did learn real anatomy: 0.318 against 0.0016 chance is 200x chance, and
  13.6x the random-init control. Supine and prone are separate acquisitions, so
  this cannot be won on noise texture.
- **Alignment and uniformity rank the encoders backwards.** MoCo is 15x more
  invariant to flip+noise than ImageNet (cosine 0.9973 vs 0.9602) at
  near-identical uniformity, so by Wang & Isola's criteria it is the better
  representation, and it loses every downstream metric. Mechanistically exact:
  flip and noise were the *only* difference between q and k, so the model drove
  that one invariance to saturation and had nothing else to optimise. Training
  worked as specified; the specification was wrong.
- **RankMe is the one geometry metric that tracked quality** (16.7 / 330 / 667
  matches the retrieval ordering). Three points is not a validation, but it is
  the cheap gate worth carrying forward.
- **The retrieval signature is not a fingerprinting signature.** Normalised by
  chance, retention of lift from `any_series` to `cross_position` is 0.20 for
  random init, 0.74 ImageNet, 0.76 MoCo. Random init is what pure
  duplicate-matching looks like; MoCo is not disproportionately duplicate-reliant.
  So the honest claim is "learned real anatomy, about a third less of it than
  ImageNet", not "learned instance fingerprints". The fingerprinting evidence
  lives in the augmentation code, the 99.7% pretext curve, and the alignment
  number — not here.
- MoCo is the **only** encoder worse than random init at depth regression
  (0.1185 vs 0.1084). Directionally consistent with same-volume keys sitting in
  the queue as negatives. Suggestive at about 2 sigma given within-volume
  correlation, not established. **Pre-registered for E4.**

## P0r — Phase 0 re-scored on the rebuilt cache (2026-09-16, job 63459067)

Same three encoders, scored on a bank rebuilt from the raw-HU cache (protocol
rule 4): 2,103 series and 33,648 crops (was 2,074 / 33,184), `knn_polyp` on the
new split, and the two confounder probes added. **These rows are the baseline
from here on.**

| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | align cos | zpos MAE | polyp bal-acc |
|---|---|---|---|---|---|---|---|
| P0r:random | 0.023 | 0.05x | 0.052 | 20.5 | 0.9991 | 0.1041 | 0.333 |
| P0r:imagenet | 0.494 | 1.00x | 0.290 | 666.9 | 0.9600 | 0.1098 | 0.349 |
| P0r:moco:checkpoint_0199.pth.tar | 0.315 | 0.64x | 0.182 | 330.3 | 0.9973 | 0.1166 | 0.356 |

Confounder probes (balanced accuracy, patient-disjoint halves, ~793 test series):

| Encoder | `knn_prep` (chance 0.333) | `knn_contrast` (chance 0.500) |
|---|---|---|
| random | 0.427 | 0.572 |
| imagenet | 0.527 | 0.538 |
| moco 0199 | 0.452 | 0.541 |

**Findings.**

- **The rebuild changed nothing that matters.** Every anatomy number moved by
  less than its standard error (cross-position 0.493 -> 0.494 and 0.318 ->
  0.315; RankMe and alignment to the third digit). The raw-HU cache, the +29
  series and HU rounding are neutral, so the P0 findings carry over intact and
  E0 onward can be compared against P0r directly.
- **Depth anomaly persists:** MoCo 0.1166 against random 0.1041 and ImageNet
  0.1098. The E4 prediction was pre-registered as "below 0.108, the random-init
  level"; on this bank the random-init level is 0.104. **Amended before E4
  runs:** the threshold is the random-init level on the bank E4 is scored on,
  0.1041. Recorded here rather than edited in place.
- **Read the confounder probes against random init, not chance.** Random
  features already score 0.427 on prep, because a random ResNet still encodes
  the intensity histogram, and prep and tagging change the histogram. ImageNet
  sits well above that (0.527) and MoCo barely (0.452), so ImageNet features
  separate prep protocols more. That is not obviously bad: cleanly tagged stool
  is real image content. It becomes a problem only if a run gains here while
  cross-position stays flat.
- **`knn_contrast` cannot rank these encoders.** The non-compliant class is
  about 30 patients per half, so resolution is several points, and all three sit
  within 0.035 of each other, with random highest. Keep logging it; watch only
  for a large move.

---

## Phase 2 ladder (planned, ACRIN-only)

Each rung is a diff against its parent. Predictions are written before
submission; results and verdicts are filled in afterwards.

### E0 — current recipe on the fast cache

- **Diff vs:** `checkpoint_0199`. Same augmentations. Three infrastructure
  changes, none a recipe choice: ACRIN-only instead of both collections; the
  rebuilt raw-HU cache instead of the `.pt` files; and per-worker seeding of the
  MONAI transforms, which previously drew identical crops and augmentations in
  all 32 workers and repeated them every epoch.
- **Why:** there is no ACRIN-only reference point. Without it, every later gain
  is confounded with dropping the pediatric collection.
- **Hypothesis:** the cache rebuild and the collection change are neutral. The
  seeding fix adds sample diversity but not a single view the pair did not
  already share, so it should not rescue the pretext task.
- **Prediction:** cross-position within about 0.03 of 0.318; alignment cosine
  still above 0.99; both kill gates fire, which is the expected and correct
  outcome for this rung.
- **Run (pre-registered 2026-09-16, before submission):** 200 epochs, not killed
  when the gates fire. E0 is the reference every rung is diffed against, so it
  needs the same epoch count as they do; the gates are recorded for it, not
  enforced. `RUN=e0 SAVE_FREQ=10`, so epoch 20 (`checkpoint_0019`) can be scored
  for the alignment gate. Scored on the rebuilt bank against the re-scored P0.
  Job 63465601. (A first submission, 63459386, was cancelled at epoch 10 for
  loading speed only; see `research_notes.md` Status B.)

### E0 — result (2026-09-17, job 63465601)

200 epochs in 4 h 50 m on 2 A100s, against ~53 h for the parent.

| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | align cos | zpos MAE | polyp bal-acc |
|---|---|---|---|---|---|---|---|
| E0:moco:checkpoint_0019.pth.tar | 0.077 | 0.16x | 0.070 | 455.8 | 0.9917 | 0.1458 | 0.368 |
| E0:moco:checkpoint_0199.pth.tar | 0.356 | 0.72x | 0.205 | 309.0 | 0.9971 | 0.1277 | 0.339 |

**Verdict: hypothesis held, prediction narrowly missed.** Predicted
cross-position within 0.03 of the parent's 0.318; E0 landed at 0.356, which is
0.041 above it, about 2.1 standard errors (SE 0.019 at n=613) before accounting
for within-patient correlation among queries, which makes the effective margin
smaller. *Corrected 2026-09-22 after audit:* the 0.041 is against P0r's 0.315
(the rebuilt-bank parent, the right baseline), and 2.1 SE treats the parent as
exact. As a difference of two runs each carrying sampling error it is about
1.5 SE, so "real" was too strong: the gain is suggestive, not established, and
the prediction may have held. Calling it a small gain at most, not a fix: E0
still reaches only 0.72x of ImageNet, and everything the parent was faulted for
is intact. Alignment cosine 0.9971 (parent 0.9973), pretext accuracy 99.9%. The
diff carried three changes at once (ACRIN-only, rebuilt cache, per-worker
seeding), so the gain cannot be attributed to one; the most likely source is
seeding, since the parent trained on ~40 distinct crop sets repeated every epoch.

- **The epoch-20 accuracy gate failed as written** and has been amended to epoch
  50 (see Kill gates). The alignment gate fired correctly at epoch 20.
- **Depth regression got worse, not better:** 0.1277 against the parent's 0.1166
  and random init's 0.1041. E0 is now the worst encoder measured on `knn_zpos`,
  which strengthens rather than weakens the same-volume-negatives story E4 tests.
- `knn_prep` fell to 0.413 from the parent's 0.452, below random init's 0.427:
  no sign that the faster pipeline made the encoder more prep-sensitive.
- The epoch-20 checkpoint is a useful midpoint: RankMe is *higher* there (455.8)
  than at the end (309.0) while retrieval is far worse (0.077), so RankMe rises
  early with feature spread and then falls as the pretext task collapses onto
  its shortcut. Read it as a floor detector, not a quality score.

---

### E1 — two independent crops

- **Diff vs:** E0. `ct_dataset.py` extracts two crops with controlled overlap
  instead of one crop deep-copied twice.
- **Why:** the single change Phase 0 points at. Currently q and k are the same
  pixels, so the task is solvable from a texture fingerprint.
- **Parameters (pre-registered 2026-09-17, before implementation):** in-plane
  overlap of the two crops is drawn uniformly from **30-70% of a crop's area**,
  split randomly between the two axes, with an independent z shift of **-2 to +2
  slices**. Rationale: at 100% overlap this is E0; at 0% the pair is often two
  unrelated organs, since a 224 mm crop of a ~356 mm volume means two disjoint
  crops can sit on opposite sides of the abdomen. 30-70% keeps a shared
  structure in both views while forcing the encoder off a pixel-exact match.
  Same crop size (224 mm) as E0, so this rung changes position only, not scale.
  Where a shift would run off the volume the crop is taken from the other side
  when that fits, and clipped to the edge otherwise, so realised overlap can
  exceed 70% on small volumes.
- **Prediction:** the largest single jump in the ladder. Acc@1 falls out of
  saturation (below 95% at epoch 50, so the amended gate does not fire);
  alignment cosine drops below 0.99; cross-position above 0.40. Also predicted:
  `knn_zpos` improves on E0's 0.1277, because a pair offset in z makes absolute
  depth less recoverable from a single crop, but it stays above random init's
  0.1041 until E4 removes same-volume negatives.

### E1 — result (trained 2026-09-17, scored 2026-09-21, job 63530506)

200 epochs in 5 h 11 m on 2 A100s, about 7% slower than E0 (4 h 50 m): two
crop reads per sample instead of one.

| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | align cos | zpos MAE | polyp bal-acc |
|---|---|---|---|---|---|---|---|
| E1:moco:checkpoint_0019.pth.tar | 0.501 | 1.01x | 0.331 | 315.3 | 0.9936 | 0.1186 | 0.332 |
| E1:moco:checkpoint_0199.pth.tar | 0.873 | 1.77x | 0.424 | 438.8 | 0.9958 | 0.0982 | 0.405 |

Pretext Acc@1 at the end of epoch index 19, 49, 99, 199 (0-based, as the log
prints them): 2.98, 37.8, 78.6, 88.3; final loss
0.829 against E0's 0.093.

**Verdict: hypothesis held, and the effect is far larger than predicted.**
Cross-position went 0.356 to 0.873, from 0.72x ImageNet to 1.77x. The
pre-registered prediction was "above 0.40"; the realised value is more than
twenty standard errors past that (SE 0.019 at n=613), so the margin survives any
plausible correction for within-patient query correlation. This is the first
encoder in the project to beat ImageNet on the bar. ~~and it does so by epoch 20
already (0.501)~~ At epoch 20 it *ties* ImageNet (0.501 vs 0.494, a third of one
SE). Phase 0's diagnosis is confirmed, on one run: q and k being the same pixels
was the defect, and drawing two overlapping crops is the fix.

Caveat on attribution: this is one run against one run. E1b (job 63806648) is
the same code and command with a different seed, and until it lands the size of
the gap is measured against unknown retrain noise. The direction is not in
doubt at this magnitude; the exact number is.

- **The alignment kill gate is invalid and is hereby withdrawn.** It predicted
  alignment cosine below 0.99. ~~E1 sits at 0.9936 at epoch 20 and 0.9958 at
  epoch 200, both *higher* than E0's 0.9917, while being better on every
  downstream measure.~~ *Corrected 2026-09-22 after audit:* like for like, E1
  is higher than E0 at epoch 20 (0.9936 vs 0.9917) and lower at epoch 200
  (0.9958 vs 0.9971), so the final cosine moved the predicted way, just not
  below 0.99. The withdrawal still stands on the gate itself: scored as written
  at epoch 20, it would have killed E1, whose epoch-20 cross-position (0.501)
  is already 6.5x E0's (0.077). A cosine above 0.99 does not indicate collapse
  here. The epoch-50 accuracy gate did work as amended: 37.8% against the 95%
  trigger that E0 hit at 94.1%.
- **RankMe's direction is not stable across runs.** E0 fell 455.8 to 309.0 while
  getting better at retrieval; E1 rose 315.3 to 438.8 while getting better at
  retrieval. Two runs, opposite signs, same outcome. Confirms the E0 reading
  that RankMe is a floor detector, and adds that it should not be read
  directionally at all.
- **Depth regression beat its predicted floor.** `knn_zpos` 0.0982, below random
  init's 0.1041, which the pre-registration said would need E4's removal of
  same-volume negatives. An independent z shift of +/-2 slices was enough. E4's
  prediction needs rewriting before E4 runs, since its stated target is already
  met. *Qualified 2026-09-22:* the margin is 0.006 with no standard error
  computed, so "met" is a point estimate. E1b gives a second reading.
- ~~**Confounders stayed flat.**~~ *Corrected 2026-09-22 after audit:*
  `knn_contrast` stayed flat (0.496 to 0.507, chance 0.500). **`knn_prep` rose**,
  0.413 (E0) to 0.490, toward ImageNet's 0.527 and well above random init's
  0.427. That is the pattern P0r flagged to watch: prep separability gaining
  alongside retrieval. It does not show the retrieval gain *is* prep sorting
  (prep is one label per patient and shared by both scans, so it could help
  retrieval only as a coarse 3-way hint, far short of 0.873), but "not the
  encoder sorting by prep" is not established either.
- `knn_polyp` 0.405 is the best measured so far (chance 0.333, E0 0.339,
  ImageNet 0.349) but n_test is 108 patients, so this is a hint, not a result.
  Phase 3 is what tests it.
- Final pretext accuracy 88.3% sits just above the 60-85% healthy band and the
  curve is still rising at epoch 200. ~~so some fingerprinting shortcut
  remains.~~ *Corrected 2026-09-22:* a rising curve does not by itself show a
  shortcut. What it does say is that the pretext task is not saturated, so E2
  and E3 have room to change it.
- **Added 2026-09-22 after audit: "the match has to come from anatomy" is
  untested.** A patient's supine and prone scans come from one session: same
  scanner, kernel, field of view and body size. `knn_collection` is 0.997, so
  the encoder carries strong acquisition features, and some of cross-position
  could be acquisition or body-size matching. ImageNet and every MoCo run are
  exposed to this equally, so rankings *between* encoders are less affected than
  the "anatomy" reading of the absolute number. Not yet ruled out by any probe.
- **Added 2026-09-22 after audit: overlap runs slightly above the stated band.**
  Simulated on the real 1,744 volume shapes, realised in-plane overlap is above
  0.70 for 7.1% of pairs (above 0.9 for 0.6%; identical pairs ~5e-6); the minimum
  is 0.298. The pre-registration allowed this on small volumes, but the cause is
  also a minor bug: `_shift_within` in `moco/__init__.py`, when neither
  direction fits, clips toward the original sign, so an edge crop shifted
  outward gets zero shift instead of the largest shift that fits. E1b runs the
  same code, so E1 vs E1b is unaffected. Fixing it changes the recipe, so it
  would be its own diff, not a silent edit.

**Audit, 2026-09-22.** A separate agent with no access to this conversation
checked the E0 and E1 write-ups against the raw logs, eval JSONs, sacct and git
history. Every tabled number, the provenance (clean tree at `1183e2c`,
`CROP_OVERLAP=0.3 0.7` per sacct) and the statistics held. Retrieval has no
self-match or duplicate leak: none of the 641 supine/prone pairs share a middle
slice, and the 613 queries come from 599 patients. The corrections above are its
interpretation findings. Pre-registration: the "above 0.40" prediction was
committed 2026-09-16 (`9e76790`), before the run. The overlap parameters landed
in the same commit as the code (`6840e76`), 8 minutes before the job started,
so "before implementation" cannot be shown from git.

**Open after E1, for the scope discussion (recorded 2026-09-22, not decided):**

1. **Headroom on the main metric.** Cross-position top-1 is 0.873 and top-5
   0.982. If E2-E5 each add a few points, they may not clear the ~0.06 threshold
   even when real. Options include keeping the metric and accepting coarser
   verdicts, or leaning harder on `any_series` and top-1 at the epoch-20
   checkpoint, where E1 sits at 0.501 with more room. Nothing changes until E1b
   prices retrain noise. *Audit note:* the worry may be overstated. Near 0.87
   the SE is about 0.013, so the two-run threshold is about 0.037, not 0.06. A
   paired test on the same 613 queries (McNemar) would be tighter still.
2. **E4's prediction is already met** (see E4). Rewrite it or drop the rung.
3. **E5 cannot be scored on cross-position as things stand.** It would train on
   the metric's own query/answer pairs (see "Where the evaluation scans come
   from"). It needs a held-out patient set, or it gets dropped.
4. **Acquisition matching.** Does part of cross-position come from scanner or
   body-size matching rather than anatomy (see the audit bullet above)? Decide
   whether a probe for it is worth adding before more rungs are judged on this
   metric.
5. **When the ladder stops.** Phase 3 needs one or two chosen recipes. E1
   alone already beats ImageNet on the proxy, so how many more rungs to run
   before Phase 3 is a scope call.

### E1b — result (trained 2026-09-22/23, job 63806648)

Same script, command and clean commit as E1 (`1183e2c`, per `git_commit.txt` in
both run dirs), `CROP_OVERLAP="0.3 0.7"`, `seed=None`, so only the random draw
differs. Ran on sg027, 20:09 to 01:46 (5 h 37 m). Scored at epochs 20 and 200
by jobs 63811030 / 63811031 (`afterok`). No pre-registered prediction beyond
"measures retrain noise": its job is to set the threshold every later rung is
judged against.

| Run | cross_pos top1 | vs ImageNet | any_series top1 | RankMe | align cos | zpos MAE | polyp bal-acc |
|---|---|---|---|---|---|---|---|
| E1b:moco:checkpoint_0019.pth.tar | 0.160 | 0.32x | 0.102 | 374.4 | 0.9900 | 0.1548 | 0.325 |
| E1b:moco:checkpoint_0199.pth.tar | 0.874 | 1.77x | 0.463 | 428.7 | 0.9954 | 0.0910 | 0.385 |

Pretext Acc@1 at epoch index 19, 49, 199: 0.73, 11.8, 86.9 (E1: 2.98, 37.8,
88.3). Final loss 0.976 (E1 0.829).

**Verdict: E1 replicates at epoch 200.** Differences E1b minus E1, final
checkpoint:

| metric | E1 | E1b | diff |
|---|---|---|---|
| cross_position top1 | 0.873 | 0.874 | +0.002 (one query of 613) |
| cross_position top5 | 0.982 | 0.976 | -0.007 |
| any_series top1 | 0.424 | 0.463 | +0.039 |
| knn_zpos MAE | 0.0982 | 0.0910 | -0.007 |
| knn_polyp bal-acc | 0.405 | 0.385 | -0.020 |
| knn_prep bal-acc | 0.490 | 0.487 | -0.003 |
| knn_contrast bal-acc | 0.507 | 0.532 | +0.025 |

- **Main metric: retrain noise is negligible next to sampling noise.** One pair
  of runs is one sample of the noise, not its spread, but a 0.002 gap against a
  sampling SE of ~0.013 at this level means the two-run threshold stays set by
  sampling (~0.04 near 0.87), not by retraining. The 1.77x-ImageNet result is
  not a lucky seed.
- **`any_series` moves ~0.04 on a retrain.** It was the proposed fallback for
  headroom (open question 1). On this evidence a later rung needs more than
  ~0.04 on it to count, before sampling noise is added.
- **`knn_polyp` moves 0.02 on a retrain**, consistent with the earlier reading
  that it cannot rank encoders. E1's 0.405 "best so far" is within retrain noise
  of 0.385.
- **Depth regression below random init replicates** (0.091 vs 0.104), so E4's
  original target is met on two runs, not one.
- **Epoch 20 does not replicate: 0.501 vs 0.160.** E1b's loss stayed near the
  uniform-guess level (ln 16385 = 9.7) far longer: 8.83 at epoch 20 against
  E1's 7.23, and 11.8% Acc@1 at epoch 50 against 37.8%. The two runs take off at
  different times and end at the same place. Consequences:
  - Early checkpoints cannot rank recipes. Open question 1's option of "leaning
    on top-1 at the epoch-20 checkpoint" is ruled out.
  - The epoch-50 Acc@1 gate still works as a saturation check (E1b's 11.8% is
    nowhere near 95%), but a low epoch-50 value is not a sign of failure.
  - Future rungs keep `SAVE_FREQ=10` so the curve can be scored at more points
    if a later run looks slow to start.

---

### E2 — scale jitter

- **Diff vs:** E1. Crop 160-320 mm FOV, resize to 224, instead of a fixed 224 mm.
- **Why:** every crop is currently exactly 224 mm, so scale invariance is never
  required. This is the closest HU-safe analogue of RandomResizedCrop.
- **Prediction:** smaller than E1 but positive; most visible on `any_series`,
  which rewards matching across reconstructions at differing effective scale.
- **Implementation, fixed before the code (2026-09-25):**
  - Positions are drawn exactly as in E1 (224 mm reference crop, 30-70%
    overlap, z shift +/-2). Each view's in-plane side is then drawn
    independently, uniform in 160-320 mm, about that view's E1 centre. So the
    diff vs E1 is scale only; realised overlap now varies with the two sizes,
    which is inherent to scale jitter.
  - A crop that would run off the volume is moved inward to fit; an axis
    shorter than the crop is taken whole and zero-padded (0 after windowing,
    what `ResizeWithPadOrCropd` already pads with).
  - The windowed crop is resized in-plane to 224 x 224, bilinear with
    antialiasing. Depth stays 3 slices at 1 mm: this is in-plane scale only.
  - Flags: `--crop-scale 160 320`, `CROP_SCALE="160 320"` in `train_moco.sh`.
    Everything else as E1: `RUN=e2 SAVE_FREQ=10 CROP_OVERLAP="0.3 0.7"`.
  - The crop bank is unchanged (fixed 224 mm), so E2 is scored on the same
    bank as E0/E1/E1b and every row stays comparable.
- **Decision rule, fixed before running (2026-09-25).** Thresholds combine
  E1b's retrain spread with sampling noise, against the E1/E1b mean (cross 0.874,
  any_series 0.444):
  - *Prediction held:* `any_series` top-1 >= 0.50 and cross-position >= 0.83.
  - *Harmful:* cross-position < 0.83. E3 then builds on E1, not E2.
  - *No detectable effect:* anything else. E3 builds on E1 (the simpler
    recipe), and E2 is recorded as a null, not a failure of the idea.

### E3 — HU window jitter

- **Diff vs:** E2. Jitter window centre and width (about +/-30 HU) per view.
- **Why:** the HU-respecting stand-in for the colour jitter MoCo v2's own
  ablations find necessary, deliberately dropped here with nothing substituted.
- **Note:** only a real experiment if the cache stores un-windowed HU. On a
  cache fixed at `[-150, 250]` this degrades to shuffling levels inside an
  already-clipped band. See the Phase 1 decision.
- **Prediction:** small positive. Largest effect on the subset that took
  iodinated oral contrast, since tagging is the intensity-dependent signal.

### E4 — exclude same-volume keys from the negatives

- **Diff vs:** E3. Mask keys drawn from the query's own volume out of the queue.
- **Why:** 41,480 samples/epoch against a 16,384 queue means the queue holds
  ~40% of an epoch, so with 20 crops/volume roughly 8 crops of the same scan sit
  in the negatives whenever the query comes from that scan. Adjacent slabs of
  one colon are being pushed apart while identical pixels are pulled together.
- **Prediction (pre-registered, from P0):** `knn_zpos` MAE drops **below 0.108**,
  the random-init level. This is the rung that tests the P0 depth-regression
  anomaly, and the prediction is falsifiable independent of the retrieval number.
- **Superseded before running (2026-09-22):** E1 already met this target
  (`knn_zpos` 0.0982, below the P0r random-init level of 0.1041) through its
  +/-2 slice z shift, without touching the queue. The mechanism in *Why* is
  still present, so E4 is not void, but it needs a new falsifiable prediction
  written before it runs, or a decision to drop it. Open; see E1's result.

### E5 — supine/prone positives

- **Diff vs:** E4. Draw the key from the same patient's other-position series.
- **Why:** 587 ACRIN patients have both. A true positive pair from two separate
  acquisitions, free of annotation.
- **Risk:** needs rough z-alignment or it pairs liver with pelvis. If it hurts,
  that is informative about how much the pair depends on spatial correspondence.
- **Prediction:** helps cross-position specifically, since it is the same task
  as the metric. Watch for it helping cross-position while leaving `any_series`
  flat, which would mean it taught the metric rather than the anatomy.

---

## Phase 3: does pretraining on unlabeled CT colonography help find polyps?

This is the claim the project exists for. Phase 2 chooses a recipe; Phase 3
tests whether it matters clinically.

### The claim, stated so it can fail

> An encoder pretrained with MoCo on unlabeled CT colonography scans detects
> polyps better than ImageNet weights or random weights, **and the gap is largest
> when few labelled patients are available.**

The second half is what makes unlabeled scans worth having. If MoCo only
matches ImageNet with all labels, pretraining bought nothing. If it beats
ImageNet with 25% of the labels, the 480 patients who have no labels did
work that labels would otherwise have to do.

### Why the linear probe as written cannot test it

Labels are per patient ("this patient has a 9 mm polyp somewhere"). A polyp is
6-10 mm inside a colon that fills a ~400-slice scan. The current probe gives a
random 224 mm crop the patient's label, and that crop almost never contains the
polyp. For nearly every crop the label is noise. That is why `knn_polyp` sits
at chance for **every** encoder, ImageNet included: no encoder can do this task,
so it cannot rank encoders. Details under "Why the lincls design as written
cannot work" in `research_notes.md`.

### The design

1. **Where the labels come from.** 345 labelled patients (242 none / 68 medium /
   35 large, after resolving TCIA's two double-listed patients) once the re-split restores the 40 `CTC-` patients. 74 of the 104
   polyp patients also give the **slice number** of the polyp in supine and
   prone. That turns a patient-level label into one tied to a location.
2. **Slice-level probe (linear, frozen encoder).** Positive crops are taken at
   the annotated polyp slice; negatives are crops from no-polyp patients at
   matched depth. The slice number gives z but not the in-plane position, so each
   polyp slice is tiled with several crops and the slice counts as positive if
   any crop fires. Metric: AUROC. This is where `main_lincls.py` comes back, with
   labels that describe the crop.
3. **Patient-level classifier (attention-MIL, frozen encoder).** Encode every
   crop in a scan, let an attention layer weight them, predict none / medium /
   large. Uses all 345 patients, including the 30 polyp patients with no slice
   number. Metrics: AUROC for polyp vs none, balanced accuracy for size.
4. **Label-efficiency curve.** Repeat 2 and 3 using 10%, 25%, 50% and 100% of the
   labelled training patients, for random init, ImageNet, and the best one or two
   Phase 2 rungs. The claim is supported if the MoCo curve sits above ImageNet
   and the gap widens as the fraction shrinks.
5. **Trust.** 104 polyp patients is too few for one fixed split: one unlucky
   split moves the answer. Use patient-level 5-fold cross-validation stratified
   by class, and report mean and spread across folds with bootstrap confidence
   intervals. A test fold's patients are excluded from MoCo pretraining for that
   fold's model, so no encoder has seen its test scans even unlabeled.
6. **Guard against tuning on the answer.** Phase 3 scores only encoders already
   chosen on the label-free Phase 2 metric. Fine-tuning the whole network comes
   after the frozen-encoder results, as the practical version of the same test.

### Open decisions for Phase 3

- How many crops tile a polyp slice, and how much depth tolerance around the
  annotated slice counts as positive.
- Whether step 5's per-fold pretraining exclusion is worth 5 pretraining runs,
  or whether one run excluding a single held-out test set is enough.

---

## Additions queued alongside the ladder

### Acquisition probe: is cross-position matching scanner and body, not anatomy?

Open question 4 after E1. Pre-registered 2026-09-25, before the code.
`scripts/eval/acquisition_probe.py`, run on the existing bank, so no encoder is
retrained and every row stays comparable.

- **Descriptor per ACRIN series**, from the first DICOM header of its raw series
  and from the cached volume: scanner (manufacturer + model), convolution
  kernel, kVp, reconstruction diameter, in-plane extent and scan length at 1 mm,
  and body cross-section area (voxels above -500 HU, mm^2) at 25/50/75% depth.
  Plus the patient's bowel prep and contrast from `clinical_data.csv`, which a
  patient's two scans always share.
- **A. Acquisition-only retrieval (no encoder).** Rank the 671 prone scans for
  each of the 613 supine queries by distance on that descriptor alone:
  z-scored continuous fields, plus a penalty of 10 per categorical mismatch.
  Top-1 is how far scanner, protocol and body size get on the main metric
  without looking at anatomy.
- **B. Look-alike gallery (per encoder).** Each supine query gets its own
  gallery: its patient's prone scan(s) plus the 20 other patients' prone scans
  nearest to it under A's distance. The encoder ranks only within that set.
  Chance is ~1/21 per query. Scored for random, ImageNet, E0, E1, E1b, and E2 when it lands.
- **How it will be read, fixed now:**
  - A top-1 >= 0.50 means acquisition and body size alone match ImageNet's
    0.494, and the "vs ImageNet" framing needs this caveat up front.
  - E1/E1b B top-1 >= 0.70: the match is not mainly acquisition or body size;
    the anatomy reading is supported as far as this descriptor reaches.
    < 0.30: not supported. In between: partly.
  - If E1's lead over ImageNet shrinks by more than half on B, the ImageNet
    comparison is confounded by acquisition and is reported that way.
- **What it cannot rule out:** anything the descriptor misses. Per-session
  noise texture (dose), table position and fine body shape are not in it. A
  high B score narrows the confound; it does not close it.

All three below were implemented 2026-09-16, before E0 (details in
`research_notes.md` Status B).

- **Contrast/prep confounder probe** in `eval_repr.py`, from
  `metadata/csv_metadata/clinical_data.csv`. Three prep protocols split the
  cohort 401/195/155 and 60 subjects did not take the iodinated contrast as
  directed. Oral contrast tags stool bright, so this is a *visible* confounder:
  an encoder that sorts by tagging rather than anatomy shows up here. Sharper
  than the scanner-make control because the effect is physical, not metadata.
- **Save args into checkpoints** (`main_moco.py`). Five of nine existing
  checkpoints have unverifiable provenance because the save path stores only
  `epoch` and `arch`. Blocking for a six-run ladder.
- **Fix the `majority_floor` print** in `eval_repr.py`: it prints a
  plain-accuracy floor beside a balanced-accuracy score, so chance-level
  `0.333 (floor 0.791)` reads as catastrophic failure.
