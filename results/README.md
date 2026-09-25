# `results/`

The data behind the numbers in the thesis, at three levels of detail.

Only the cells the dissertation reports are here. A fair number of runs were
discarded along the way, and none of them made it into these files: the
grounded reader (`_v2`), `first_results`, `cognee_fixed`, the HippoRAG 2 index
built in batches, the `_rephrased` variants, and every smoke run and gate.

Column names are in Portuguese, the language the study was run in. The prose
here is in English, like the dissertation.

## How the study is laid out

Two datasets, MuSiQue-Ans and 2WikiMultiHopQA, a thousand questions each. Two
arms: in the *controlled* arm every framework hands its documents to the same
fixed reader, and in the *native* arm each framework runs its own pipeline end
to end. Four frameworks: LightRAG, Microsoft GraphRAG, HippoRAG 2 and Cognee.

That is 16 cells. Add the three controls that exist only in the controlled arm,
the closed-book floor, the dense baseline and the oracle ceiling, across both
datasets, and you arrive at the 22 cells in the aggregate files.

## The files

| file | rows | one row is |
|---|---:|---|
| `aggregate/arm_a_controlled.csv` | 14 | a cell of the controlled arm |
| `aggregate/arm_b_native.csv` | 8 | a cell of the native arm |
| `per_question/arm_a_controlled.csv` | 14,000 | one question, answered by one cell |
| `per_question/arm_b_native.csv` | 8,000 | the same, native arm |
| `retrieval/arm_a_documents.csv` | 96,599 | one retrieved document |
| `retrieval/cognee_blocks.csv` | 2,000 | one block of text Cognee returned |
| `corpus/documents.csv` | 17,634 | one source document |

Join them on `celula`, `question_id` and `document_id`. Document text lives in
`corpus/` and nowhere else. Carrying it inside every retrieved item would have
turned 8.5 MB of text into 65 MB, and the extra 57 MB would have been the same
paragraphs over and over.

## `aggregate/`

The two tables from the results chapter, plus the columns that did not fit on
the page.

`celula` names the run and `framework`, `papel` and `dataset` place it in the
design. Then the judge's raw tally: `n` answers, `n_pontuaveis` after the
`gold_suspect` ones are set aside, and the four verdict counts, `correct`,
`partial`, `incorrect` and `refusal`. From those come `estrita` (correct over
scoreable), `lenient` (partial credit at half a point), `recusa` over all 1,000
answers, `cobertura` as its complement, and `seletiva`, which is strict
accuracy among the questions the system agreed to answer. `acordo_passes` is
how often the judge's two passes landed on the same verdict.

The controlled file then describes what the reader was given: `docs_q`
documents per question on average, `recall_5` and `all_gold_5` at the cutoff,
and `recall_pool` over everything retrieved. Where a pool runs deeper than five,
the gap between `recall_5` and `recall_pool` is evidence that was found but
ranked too low to be read.

The last columns hold the paired test. In the controlled file,
`delta_vs_denso_pp` compares each substrate to the dense baseline. In the
native file, `delta_vs_controlado_pp` compares each framework to its own
controlled counterpart, named in `celula_controlada`. Both carry `p_holm`,
`significativo` and `n_pareado`. The deltas are computed over the intersection
of scoreable items, so subtracting the two accuracy columns will not always
reproduce them.

## `per_question/`

One row per question per cell, for reading what a system actually said.

`pergunta`, `resposta_ouro` and `aliases_ouro` are the item. `resposta` is the
answer in full, exactly as it was produced. `hops` (MuSiQue) and `tipo`
(2Wiki) mark the stratum, so the files can be cut by reasoning depth or
question type.

The judge occupies the middle of the row. `label_juiz` is the verdict,
`estrita` its binary form, `valor_juiz` the score that feeds the averages.
`pass_a` and `pass_b` are the two independent passes and `acordo_passes` says
whether they agreed, with `baixa_confianca` flagging the ones the judge itself
was unsure about. `ouro_correspondente` records which gold answer it matched
and `justificacao_juiz` why. Rows marked `gold_suspect` come through with these
columns empty: those are the ones where the judge flagged the reference answer
as doubtful, which took them out of the scoreable base.

The controlled file closes with the retrieval: `docs_recuperados` is the ranked
list of document identifiers the reader received, pipe separated, alongside
`docs_ouro` and the per-question `recall_5`, `all_gold_5` and `recall_pool`.

## `retrieval/`

The ranked list opened up, one document per row, for seeing where the evidence
landed rather than whether it arrived. `rank` is the position, `score` the
retriever's own number where it persisted one, `e_ouro` marks the gold
documents, and `chunk_id` survives for the frameworks that retrieve passages
rather than whole documents.

`cognee_blocks.csv` is a separate file for a separate shape, explained below.

## `corpus/`

Every source document once: `document_id`, `titulo`, `texto`. This is what the
identifiers in the other files point to, and it is the only place the document
text appears.

## Three things worth knowing first

**The native arm has no retrieval to show.** Those eight cells answer inside
the framework and never expose the list of documents they used, so the
retrieval columns are absent there and the arm does not appear in
`retrieval/` at all. The closed-book control has no retrieval either, by
design.

**Cognee does not return a ranked list.** It returns a block of text, and the
document identifiers are pulled back out of it afterwards. Its `@k` metrics
carry that caveat, and the block itself sits in
`retrieval/cognee_blocks.csv`, which is the only faithful record of what
reached the reader.

**The dense baseline on MuSiQue was run twice.** The thesis measures its
retrieval on the twin cell `v2_grounded`, which shares the retriever and
changes only the reader. The two runs disagree about the top five on 8 of the
1,000 questions, nearly all of them ties that came back in a different order,
and on one of those the disagreement flips `all_gold@5`. So the aggregate says
0.218, as the thesis does, while the per-question file recomputes on the
`v1free` cell itself and says 0.217. Both runs round to 0.553 on `recall@5`,
and no statistical test in the thesis uses either number.

One last warning about reading these files. Answers and judge rationales
contain line breaks inside quoted fields, so `wc -l` will overcount. Any real
CSV reader, including a spreadsheet, gets the counts in the table above.
