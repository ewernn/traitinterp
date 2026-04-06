# Extra Analyses (run in parallel with subagents)

All of these are independent — spawn them all as parallel subagents.

## 1. Random trait baseline control (MOST IMPORTANT)

This answers: "would any random set of traits detect reward hacking?"

Using the onset_detector.py infrastructure:
- Build 20 templates, each from 13 RANDOM traits (sampled from all ~150 available traits)
- Run each template on the same exploit data (rm_syco/train_100)
- Compare detection accuracy (within ±10 tokens) to the real "combined" preset
- Report: mean TPR across 20 random templates vs the real 89% TPR
- If random baseline gives e.g. 30% TPR, that proves trait selection matters

To get the full trait list, look at what's available in:
`experiments/rm_syco/inference/rm_lora/projections/emotion_set/` — each subdirectory is a trait.

You'll need to modify onset_detector.py or write a wrapper that:
1. Lists all available traits
2. Randomly samples 13
3. Builds template + runs detection
4. Repeats 20 times
5. Reports mean/std TPR

Also try: 13 random traits on the BENIGN data. If random templates also give high scores on benign, then the real template's advantage is better separation, not just higher exploit scores.

## 2. Shuffled template control

Same 13 traits from "combined" preset, but shuffle the temporal axis of each trait independently (randomize the 21-token window order). This tests: does the TEMPORAL SHAPE matter, or just which traits are in the template?

Run 20 shuffled versions, report mean TPR vs unshuffled 89%.

## 3. ROC curve

Plot a proper ROC curve: exploit scores vs benign scores (using whatever benign data is available — the 30 ood_bias_eval + any new benign from the hour plan).

Save as PNG. This is a figure we can put in the 2-pager if it looks good.

## 4. Per-bias detector breakdown

The template is built from 9 biases. There are 52 total. Break down the 89% TPR by:
- Detection rate on the 9 template biases (in-distribution)
- Detection rate on the 43 held-out biases (OOD)
- Detection rate per individual bias (which biases are easy/hard to detect?)

The annotations are at `experiments/rm_syco/rm_sycophancy/analysis/bias_exploitation_annotations.json`.
The bias names are at `datasets/traits/archive/rm_hack/biases.json`.

## 5. Aria-side convolution analysis

We have onset dynamics data for Aria at `experiments/aria_rl/analysis/`. 
- Load `hack_onset_aligned_s1.npz` or the trajectory data
- Build a template from Aria data (all ~150 traits, pick top discriminative)
- Run detection on Aria test data
- Report TPR as a standalone result (not cross-model, just same-model Aria)

This gives us a convolution detector result for BOTH environments independently.

## 6. Single-trait AUC sweep

For EACH of the ~150 traits individually, compute: how well does that single trait's onset shift discriminate RH from baseline responses? (simple AUC on the onset shift magnitude)

Report top-20 and bottom-20 traits by AUC. This shows which traits carry signal and which don't — directly addresses "would any trait work?"

## When done
Write all results and key numbers to `experiments/rm_syco/rm_sycophancy/analysis/extra_analysis_findings.md`
Save all figures to `experiments/rm_syco/rm_sycophancy/analysis/`
Push to R2 and git.
