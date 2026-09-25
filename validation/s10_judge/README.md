# S10 — Human validation of the LLM judge (gpt-4o, judge_v1)

**The spreadsheets in this folder were filled in by hand, by the author.** They
are the human side of the agreement study, and they came out of the original
experimental run. Nothing in this package regenerates them — running the
pipeline produces judge labels, never the human ones. Treat them as evidence,
not as output.

What S10 committed to: validate the judge against 80–120 human-labelled items
and report percent agreement plus Cohen's κ. This folder holds **120 items**,
60 per dataset, drawn from the 9 judged cells of each one — four frameworks
across two arms, plus the dense baseline. Seed 42, and the natural label
distribution, with no balancing. Balancing would have distorted the κ.

## What is here

| | |
|---|---|
| `{musique,twowiki}/s10_*.xlsx` | **the hand annotation** — the `human_evaluation` column is the author's own judgement, row by row |
| `keys/judge_key_*.csv` | the judge's original labels, never edited. The join key |
| `keys/sampling_manifest*.json` | how the 120 items were drawn |
| `s10_results.json` | the published agreement and κ |

The xlsx carry no annotator identity: `dc:creator` is empty or `openpyxl`,
`cp:lastModifiedBy` is empty, and there is a test that keeps it that way,
because a spreadsheet stores authorship in `docProps/core.xml` and it would
travel without anyone noticing.

## How the review was done

1. Open `musique/s10_musique_human_validation.xlsx` and
   `twowiki/s10_twowiki_human_validation.xlsx`.
2. Row by row: read `question`, `gold_answer` (plus `gold_aliases`) and
   `ai_answer`, then decide the correct label.
3. The **`human_evaluation` column, the yellow one, arrives pre-filled with the
   judge's own label** — change it from the dropdown only where you disagree.
   `notes` is free text.
4. **Leave the other columns alone.** They are the join key.

## The labels

These are the definitions from the judge's prompt.

- **correct** — the prediction ASSERTS the gold answer, or something plainly
  equivalent: a synonym, an abbreviation, an alternative name, a different date
  or number format, the same entity by another route.
- **partial** — a multi-hop chain that got part of the way. The intermediate
  step is right but the final answer is wrong or incomplete; or the answer
  hesitates between candidates and the right one is among them; or it is correct
  but less specific than the gold.
- **incorrect** — commits to a final answer that is WRONG: wrong entity, wrong
  value, or an answer to the wrong sub-question.
- **refusal** — declines, says there is not enough information, denies the
  premise, or comes back empty.
- **gold_suspect** — the gold answer or its aliases look wrong, ambiguous or
  broken, and the item cannot be judged fairly.

Two rules inherited from the judge: verbosity, markdown and preamble NEVER cost
anything, and what is required is ASSERTION of the gold answer, not a passing
mention of it.

## After the review

Join the reviewed xlsx against `keys/judge_key_*.csv` — the judge's original
labels, which never change — on `question_id` and `cell`. That gives percent
agreement and Cohen's κ, both over the five classes and over the strict binary
split of correct against everything else. The sampling parameters (seed 42,
per-cell quotas, sources) are recorded in `keys/sampling_manifest.json`.

`scripts/compute_judge_agreement.py` does this and reproduces the published
figures from the raw labels.

## A methodological note, and it belongs in the thesis

The column arrives pre-filled with the judge's label, so this is a protocol of
**human verification and correction**, not blind independent labelling. That
carries an anchoring risk, and anchoring tends to inflate agreement. The design
should be reported the way it was actually run.
