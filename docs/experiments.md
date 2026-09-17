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

Stop a pretraining run early if either fires. Both mean the pretext task is too
easy, which is the failure Phase 0 diagnosed, and neither needs the full run to
detect:

- `Acc@1` still above **95% at epoch 50**. Healthy contrastive training sits
  around 60-85%. **Amended 2026-09-17 after E0**, which sat at 0.4% at epoch 20
  (chance is 0.39%) and still reached 99.9%: with per-worker seeding the curve
  now takes off between epochs 20 and 49, so the epoch-20 reading says nothing.
  E0's numbers by epoch: 0.4 (20), 94.1 (49), 99.8 (99), 99.9 (199).
- Augmented-view cosine (`alignment_cos`) above **0.99**, scored on the
  epoch-20 checkpoint. The collapsed checkpoint sits at 0.9973 against
  ImageNet's 0.9602. This one did fire on E0 at epoch 20 (0.9917) while the
  accuracy gate was silent, so of the two it is the earlier signal.

### The bar

**ImageNet cross-position retrieval top-1 = 0.494** (P0r, the rebuilt bank; was
0.493 on the P0 bank). Label-free, ACRIN-only, n=613 queries, standard error
about 0.02, so it resolves a 5-point change. Until a run beats that, there is no
result. Chance is 0.0016.

Secondary, and not difficulty-matched to each other: `any_series` top-1 (same
task over the full ACRIN series pool, where alternate reconstructions of one
acquisition make it winnable by near-duplicate matching) and RankMe.

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

**The main number.** Take a patient's supine scan and ask which of the ~600
prone scans is most similar. Correct means it picks the same patient's prone
scan. `cross_position_top1` is the fraction of 601 supine scans that get it
right. Between the two scans the patient rolled over, so gas, fluid and
collapsed bowel have all moved and the pixels do not line up. What stays the
same is the patient's anatomy, so a model can only score well by encoding
anatomy. Chance is 0.16%, ImageNet gets 49%, the old MoCo 32%.

**Why it can be trusted:**

- *Same input for everyone.* One fixed crop bank, so a difference comes from
  the encoder, not from which crops it happened to get.
- *Two controls scored identically.* Random init shows what the architecture
  gives for free; ImageNet is the bar a CT-specific pretraining has to clear.
- *Known sampling noise.* With 601 queries the standard error is about 0.02, so
  two runs have to differ by about **0.06** before the gap is more than noise.
- *Predictions written first.* Each rung states its expected result before it
  runs, so a result cannot be explained after the fact.
- *Cross-checks that disagree for known reasons.* Kill gates and alignment
  catch a too-easy pretext task, RankMe catches collapse, and the confounder
  probe catches an encoder sorting scans by bowel prep instead of anatomy.

**What it does not tell us:**

- *Retraining noise is unmeasured.* The 0.02 covers which scans were queried,
  not what a retrain with a different seed would score. Training E1 twice
  would measure that. If the two differ by more than 0.06, every threshold
  above has to widen.
- *It is a proxy for anatomy, not for polyps.* A model can recognise a patient
  without seeing a polyp. The ladder uses this number to *choose a recipe*
  because it is label-free and has 601 queries. Whether that recipe helps find
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
smaller. Calling it a real but small gain, not neutrality, and not a fix: E0
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
- **Prediction:** the largest single jump in the ladder. Acc@1 falls out of
  saturation; alignment cosine drops below 0.99; cross-position above 0.40.

### E2 — scale jitter

- **Diff vs:** E1. Crop 160-320 mm FOV, resize to 224, instead of a fixed 224 mm.
- **Why:** every crop is currently exactly 224 mm, so scale invariance is never
  required. This is the closest HU-safe analogue of RandomResizedCrop.
- **Prediction:** smaller than E1 but positive; most visible on `any_series`,
  which rewards matching across reconstructions at differing effective scale.

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
