# Evaluating automated annotation of *Pride and Prejudice*: versions 3–6 against a human reference

## Abstract

This study compares four automatically annotated XML editions of *Pride and Prejudice* with the supplied human-annotated workbook, across seven hierarchically prioritised objectives: turn detection, speaker attribution, addressee attribution, listener presence, listener identity, indirect-representation detection, and indirect-representation subtype. The analysis combines full-file computational comparison with targeted close reading of disagreements. XML v1 and v2 are excluded from evaluation, as requested.

The latest output, XML v6, achieves the strongest agreement on recoverable turn spans and named listeners. Speaker attribution is consistently strong on matched turns, whereas audience reconstruction remains substantially more difficult. These findings do not establish uniformly improving literary accuracy: the human reference contains incomplete span marking, ambiguous participant decisions, and at least one demonstrable attribution error. Moreover, the latest pipeline changes the treatment of letters, self-address, and reported thought. Accordingly, the results below distinguish reference agreement, conditional attribution performance, end-to-end performance, and changes in annotation scope. No reliable corpus-wide IS-versus-FID accuracy can be estimated from the available subtype labels.

## 1. Materials, provenance, and research questions

### 1.1 Evaluated outputs and generating scripts

Paths below assume that the scripts, XML files, and workbook are stored two directory levels above this report. Actual supplied script filenames contain a hyphen before the version, rather than the underscore used in the task description.


| Output | Generating script | Status |
| --- | --- | --- |
| [XML v3](../../pride_and_prejudice_annotated_v3.xml) | [annotate_austen-v7.py](../../annotate_austen-v7.py) | Evaluated |
| [XML v4](../../pride_and_prejudice_annotated_v4.xml) | [annotate_austen-v10.py](../../annotate_austen-v10.py) | Evaluated |
| [XML v5](../../pride_and_prejudice_annotated_v5.xml) | [annotate_austen-v11.py](../../annotate_austen-v11.py) | Evaluated |
| [XML v6](../../pride_and_prejudice_annotated_v6.xml) | [annotate_austen-v16_1.py](../../annotate_austen-v16_1.py) | Evaluated |

The human reference is [pride_and_prejudice_JK.xlsx](../../pride_and_prejudice_JK.xlsx), worksheet `Tabelle1`. Historical provenance supplied by the researcher is XML v1 ← script v3 and XML v2 ← script v4. Those two outputs and scripts are not evaluated here. Script-to-output attribution follows the researcher's account; XML metadata alone does not independently prove which script or command produced an output.

All four XML roots identify the model as `gemini-4-argon`. This is a recorded identifier, not an independently verified provider specification. Runtime command lines, complete checkpoints, usage logs, and repeated runs were not supplied. Script defaults therefore describe intended behaviour, not confirmed settings for each historical run.

### 1.2 Objectives and priorities


| Objective | Priority | Operational evaluation |
| --- | --- | --- |
| 1. Turn detection | Must | One-to-one span matching; exact-boundary sensitivity |
| 2. Speaker identity | Must | Canonical identity-set precision, recall, F1; exact-set accuracy |
| 3. Addressee identity | Must | Canonical identity-set precision, recall, F1; exact-set accuracy |
| 4. Listener presence | Must | Binary confusion matrix on matched turns |
| 5. Active listener identity | Should | Canonical listener-set precision, recall, F1; interpretive audit |
| 6. Indirect representation present | Should | Matched-span classification plus positive-row coverage |
| 7. Indirect representation subtype | Could | Label inventory and explicitly qualified subtype audit |

## 2. Evaluation methodology

### 2.1 Reconstructing the human reference

The workbook contains line-wrapped source text and columns for speaker, addressee, bystander, indirect speech, and comments. `$` marks a turn opening and `$$` a closing. Text rows are grouped across consecutive nonblank rows. Within each group, complete opening–closing pairs define reference spans. When the number of speaker-bearing metadata rows equals the number of spans, metadata are assigned in textual order. This handles cases where metadata appear before the actual opening marker. Otherwise, metadata are assigned only from rows overlapping the span, and ambiguous assignments are excluded from attribute evaluation.

This conservative procedure recovers **1,383 complete marked spans**. Three have ambiguous metadata assignment, at worksheet rows 6180, 7210, and 12424. They remain in detection evaluation but not attribute scoring. There are **18 marker anomaly events**, not necessarily 18 distinct erroneous turns. These include orphan closings, repeated openings, and openings not closed within a nonblank group. A span crossing a blank row can therefore be excluded even if a reader could reconstruct it. The procedure does not silently repair these cases.

Consequently, the detection benchmark is the **recoverable marked-span subset**, not a claim that the human annotator intended exactly 1,383 turns. Unmarked or malformed reference passages may appear as false positives when the model correctly annotates them. A manually adjudicated reference is required before interpreting these counts as absolute accuracy.

### 2.2 Text alignment and boundaries

All files are read in full. XML text is reconstructed recursively, retaining turn and letter offsets and preserving nested-turn records. Text is normalised to lowercase alphanumeric characters only. Whitespace, punctuation, underscores, and annotation markers do not contribute to offsets. This removes line wrapping and typographical variation, but also means that “exact” below is **exact in normalised alphanumeric coordinates**, not byte-exact or punctuation-exact.

Each XML text is aligned to the workbook text with a minimum-edit character alignment. Boundaries in equal blocks map directly; a boundary within an edited region maps to the nearest aligned boundary. This fallback is a limitation around textual variants. All four XML texts contain 553,748 normalised characters; the workbook contains 553,768. Each alignment contains 553,576 equal characters and an edit distance of 303. Thus the texts are closely aligned, but not literally identical. Front and back matter are included in alignment; detection scores concern annotated spans rather than unannotated boilerplate.

### 2.3 Matching and metric definitions

For each reference–prediction pair, intersection-over-union is computed as:

`IoU = overlapping normalised characters / characters in the union of the two spans`.

A maximum-total-IoU one-to-one assignment is computed, after which pairs below **IoU 0.50** are discarded. Unmatched predictions count as FP and unmatched reference spans as FN. This is a relaxed boundary criterion: substantial overlap can count as a detection even when the model includes an attribution clause or merges adjacent content. Exact-normalised-span scores are reported separately. No matching is based on turn numbering, which changes between versions.

`Precision = TP / (TP + FP)`; `Recall = TP / (TP + FN)`; `F1 = 2PR / (P + R)`.

For participant sets, TP, FP, and FN are identity memberships, micro-aggregated across eligible matched turns. Exact-set accuracy requires the entire predicted set to equal the reference set. Empty listener sets contribute to exact-set accuracy but not positive membership TP. Conditional metrics evaluate attribution only after a turn has been matched; they must not be read as whole-pipeline accuracy. Supplementary end-to-end metrics penalise unmatched predictions and missed reference spans.

### 2.4 Identity normalisation and exclusions

Names are split on `+`, stripped, case-normalised, and treated as unordered sets. A fixed alias mapping is extracted from the supplied v7 script and applied identically to every version. Explicit additions reconcile `Charlotte`/`Mrs. Collins`, `Miss De Bourgh`/`Miss Anne de Bourgh`, `a young Lucas`/`Master Lucas`, and `Catherine`/`Kitty`. The pipeline's unusual canonical label `Mrs. Elizabeth Bennet` is treated as Elizabeth's identity, not as a historically appropriate title throughout the novel.

The mapping is not inferred from which answer would maximise agreement. Nevertheless, using a script-derived registry can favour its naming conventions. Raw-name exact-set accuracy is therefore also reported. Ambiguous bare surnames and collective labels remain interpretive risks. A canonical match does not prove that the underlying attribution is correct.

Missing human speaker/addressee labels are excluded from the corresponding identity metric. Human `?` is treated as unresolved and excluded, not as a definite absence. `the butler` and `one of the girls` are mapped to unresolved identity for scoring and excluded from identity evaluation. Local role references in v6 are discussed separately. Blank human bystander cells are provisionally treated as “none”; this is necessary for binary scoring but may confuse missing annotation with actual absence. Human `?` bystander cells are excluded. A predicted `Unknown` listener is counted as presence, but cannot match a named listener.

### 2.5 Indirect speech and missing subtype labels

The workbook has **244 nonempty indirect-speech flags**: 85 `yes`, 158 `x`, and one `xFID`. Only 80 `yes` flags attach to the recovered marked spans under the metadata procedure. The remaining row-level evidence must not be discarded or conflated with those 80 span labels.

Two complementary analyses are therefore reported:

1. Conditional binary classification on recoverable matched spans, treating `IS` and `FID` as positive. Blank span labels are provisionally negative.
2. Coverage of all 244 positive workbook rows: a hit occurs when any model `IS`/`FID` span overlaps at least one normalised character of that row. This is **positive-row coverage**, not turn recall or precision. Row labels can refer to adjacent text, and minimal overlap is permissive.

The generic labels `yes` and `x` are not converted into IS subtype labels. There is only one explicitly subtype-marked row. Consequently, multiclass IS/FID precision, recall, macro-F1, or accuracy would be scientifically misleading. The report does not invent them.

## 3. Structural and scope audit


| XML | Turns | NO | IS | FID | Separate letters | Nested turns |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1749 | 1378 | 271 | 100 | 0 | 6 |
| v4 | 1748 | 1361 | 315 | 72 | 0 | 0 |
| v5 | 1805 | 1329 | 384 | 92 | 0 | 0 |
| v6 | 1655 | 1320 | 266 | 69 | 63 | 0 |

XML v3 contains six nested turns. XML v4–v6 contain none, consistent with the flat-turn validation introduced in script v10. This is a directly observable structural improvement, not proof of better semantic attribution.

XML v6 contains 63 separate `<letter>` elements. These are excluded from the primary `<turn>` comparison. Earlier versions can encode letter content as turns, so raw turn counts and detection precision are not fully scope-equivalent. Simply adding 63 letters to the v6 turn count would not solve the problem: letter segmentation and spoken reading are not interchangeable with conversational turns. The workbook's letter comments are not an exhaustive independent letter-span gold standard. A fully harmonised spoken-only or all-representation benchmark requires adjudication.

## 4. Quantitative results

All percentages are rounded to two decimals. Counts are unrounded. “FP” means disagreement with the operational reference, not necessarily a literary error.

### 4.1 Objective 1: turn detection (Must)

#### Relaxed matching, IoU ≥ 0.50

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1358 | 391 | 25 | 77.64 | 98.19 | 86.72 |
| v4 | 1348 | 400 | 35 | 77.12 | 97.47 | 86.11 |
| v5 | 1320 | 485 | 63 | 73.13 | 95.44 | 82.81 |
| v6 | 1365 | 290 | 18 | 82.48 | 98.70 | 89.86 |

#### Exact normalised boundaries

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1296 | 453 | 87 | 74.10 | 93.71 | 82.76 |
| v4 | 1286 | 462 | 97 | 73.57 | 92.99 | 82.15 |
| v5 | 1235 | 570 | 148 | 68.42 | 89.30 | 77.48 |
| v6 | 1295 | 360 | 88 | 78.25 | 93.64 | 85.25 |

V6 has the highest relaxed and exact-span F1. V5 has the lowest span agreement despite producing the largest number of turns. This is compatible with more expansive indirect-representation annotation and changes in splitting, but it does not isolate a causal mechanism. Incomplete human marking and letter-policy changes affect the apparent precision penalty.

### 4.2 Objective 2: speaker identification (Must)

Conditional canonical identity-membership metrics:

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1360 | 3 | 3 | 99.78 | 99.78 | 99.78 |
| v4 | 1347 | 3 | 5 | 99.78 | 99.63 | 99.70 |
| v5 | 1318 | 3 | 6 | 99.77 | 99.55 | 99.66 |
| v6 | 1365 | 3 | 5 | 99.78 | 99.64 | 99.71 |

| XML | Eligible matched turns | Canonical exact-set accuracy (%) | Raw-name exact-set accuracy (%) |
| --- | --- | --- | --- |
| v3 | 1351 | 99.78 | 25.68 |
| v4 | 1341 | 99.70 | 25.80 |
| v5 | 1313 | 99.62 | 25.06 |
| v6 | 1359 | 99.71 | 25.75 |

Speaker agreement is near ceiling on matched, identity-resolved reference turns. Small differences between versions should not be overinterpreted: some disagreements are demonstrable reference errors, some involve collective speakers, and conditional denominators differ. High speaker F1 does not imply equally high turn detection or audience reconstruction.

### 4.3 Objective 3: addressee identification (Must)

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1493 | 72 | 36 | 95.40 | 97.65 | 96.51 |
| v4 | 1430 | 46 | 78 | 96.88 | 94.83 | 95.84 |
| v5 | 1396 | 59 | 85 | 95.95 | 94.26 | 95.10 |
| v6 | 1448 | 37 | 85 | 97.51 | 94.46 | 95.96 |

| XML | Eligible matched turns | Canonical exact-set accuracy (%) | Raw-name exact-set accuracy (%) |
| --- | --- | --- | --- |
| v3 | 1328 | 95.93 | 18.90 |
| v4 | 1318 | 94.99 | 19.04 |
| v5 | 1290 | 93.49 | 18.53 |
| v6 | 1336 | 95.28 | 19.09 |

Addressee identification is less stable than speaker attribution. V3 has the highest conditional membership F1 in this benchmark. V6 does not dominate every Must objective. Multiple recipients, a shift from one addressee to another within a paragraph, and the distinction between addressee and overhearer all create genuine interpretive disagreements. Empty self-address in v6 also differs from earlier versions' self-as-addressee convention.

### 4.4 Objective 4: presence of listeners/bystanders (Must)

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 536 | 123 | 18 | 81.34 | 96.75 | 88.38 |
| v4 | 530 | 99 | 17 | 84.26 | 96.89 | 90.14 |
| v5 | 499 | 85 | 18 | 85.45 | 96.52 | 90.64 |
| v6 | 538 | 59 | 19 | 90.12 | 96.59 | 93.24 |

| XML | TN | Accuracy (%) | Eligible matched turns |
| --- | --- | --- | --- |
| v3 | 658 | 89.44 | 1335 |
| v4 | 679 | 91.25 | 1325 |
| v5 | 694 | 92.05 | 1296 |
| v6 | 726 | 94.19 | 1342 |

V6 provides the strongest binary listener-presence agreement. This measure is less demanding than identifying every listener: predicting one correct or incorrect listener can still yield a correct presence decision. Blank reference cells remain a significant source of uncertainty.

### 4.5 Objective 5: identity of active listeners (Should)

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 1755 | 1047 | 171 | 62.63 | 91.12 | 74.24 |
| v4 | 1578 | 639 | 324 | 71.18 | 82.97 | 76.62 |
| v5 | 1457 | 552 | 342 | 72.52 | 80.99 | 76.52 |
| v6 | 1600 | 280 | 333 | 85.11 | 82.77 | 83.92 |

| XML | Eligible matched turns | Exact listener-set accuracy (%) |
| --- | --- | --- |
| v3 | 1335 | 75.73 |
| v4 | 1325 | 74.34 |
| v5 | 1296 | 75.08 |
| v6 | 1342 | 81.37 |

The latest output improves listener membership F1 substantially. However, “active listening” is not directly observable from a participant list. The XML field and scene-state machinery can encode physical presence or inferred audibility; the workbook does not independently verify sustained attention. These scores therefore measure agreement on **annotated listener identities**, not validated cognitive attention.

#### Fixed-cohort sensitivity analysis

To reduce changing-denominator effects, the following scores use the 1,307 reference spans matched in all four versions. Field-specific exclusions still apply.


| XML | Speaker micro-F1 (%) | Addressee micro-F1 (%) | Listener micro-F1 (%) |
| --- | --- | --- | --- |
| v3 | 99.77 | 96.82 | 73.51 |
| v4 | 99.81 | 96.15 | 76.32 |
| v5 | 99.77 | 95.35 | 76.56 |
| v6 | 99.81 | 96.24 | 83.73 |

The listener improvement persists on this common cohort, making it less likely to be solely an artefact of which turns were matched. It remains an observational comparison of single runs, not an ablation experiment.

#### End-to-end identity sensitivity

These micro-F1 scores additionally penalise participant memberships in unmatched predicted turns and missed eligible reference spans. Matched turns with unresolved or missing reference labels are excluded for that field. Unmatched predictions remain penalised even when they correspond to unmarked human content; thus these are deliberately reference-dependent, scope-sensitive measures.


| XML | Speaker end-to-end F1 (%) | Addressee end-to-end F1 (%) | Listener end-to-end F1 (%) |
| --- | --- | --- | --- |
| v3 | 85.67 | 78.00 | 67.77 |
| v4 | 84.93 | 77.19 | 67.81 |
| v5 | 81.58 | 73.90 | 66.93 |
| v6 | 89.01 | 83.08 | 76.35 |

### 4.6 Objective 6: indirect representation detection (Should)

#### Conditional classification on recovered matched spans

| XML | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | 73 | 2 | 0 | 97.33 | 100.00 | 98.65 |
| v4 | 71 | 2 | 0 | 97.26 | 100.00 | 98.61 |
| v5 | 76 | 2 | 0 | 97.44 | 100.00 | 98.70 |
| v6 | 79 | 2 | 0 | 97.53 | 100.00 | 98.75 |

These very high conditional scores are **not whole-novel indirect-speech performance**. They primarily reflect the small, explicitly delimited `yes` subset. They omit much of the `x` evidence and condition on successful turn matching. They should not be used as the headline result for this objective.

#### Coverage of all positive workbook rows


| XML | All flags hit / 244 | Coverage (%) | yes hit / 85 | x hit / 158 | xFID hit / 1 |
| --- | --- | --- | --- | --- | --- |
| v3 | 131 | 53.69 | 83 | 48 | 0 |
| v4 | 139 | 56.97 | 84 | 55 | 0 |
| v5 | 132 | 54.10 | 84 | 48 | 0 |
| v6 | 127 | 52.05 | 84 | 43 | 0 |

Coverage of `yes` rows is high, whereas coverage of `x` rows is much lower. V4 has the highest overall positive-row coverage; v6 has the lowest. This is an important counterweight to the stronger v6 turn/listener results. It may partly reflect the later exclusion of silent reported thought, but the aggregate does not prove that explanation. The meaning and scope of `x` require annotation-guideline clarification.

No defensible whole-corpus indirect-speech precision can be inferred from positive-row flags alone: a model annotation outside a flagged row is not necessarily false, and a flag does not specify exact boundaries. The conditional precision above and row coverage answer different questions and must not be combined into one F1 score.

### 4.7 Objective 7: precise indirect-representation type (Could)

The model subtype inventories are shown in Section 3. The human workbook contains only one explicit `xFID` row, worksheet row **12707** (source line 12706). None of the four models places an IS/FID span over that row under the row-overlap rule: coverage is 0/1 in every version. This is not an estimate of FID recall.

The flag is located on the continuation row after a marked quoted question about Bingley and Jane. It may refer backwards to that question or forwards to the subsequent narration. The quoted question itself is treated as `NO` in v6. Without adjudicating the flag's intended scope, even this single subtype disagreement cannot be classified confidently as a model error.

**Conclusion for Objective 7:** quantitative multiclass evaluation is not identifiable from the supplied reference. Counts, the 0/1 explicit-row observation, and qualitative cases are the available evidence. A subtype-labelled adjudicated sample is required.

## 5. Qualitative error analysis and reference audit

Examples below are purposively selected to expose different failure mechanisms. They are not a random sample and cannot establish error-category prevalence. Worksheet row numbers include the header; source line numbers in column A are generally one lower. XML paragraph indices are zero-based, as used by the scripts. Turn numbers are version-specific. The workbook and XML links in Section 1 are the primary sources for every example.

### 5.1 A clear human speaker error: Elizabeth's short answer

At worksheet row **4459**, the narrative explicitly identifies Elizabeth as the person answering. The human speaker cell instead names Charlotte and the addressee cell names Elizabeth. V6, paragraph **787**, turn **687**, assigns Elizabeth as speaker. V3 and v5 also assign Elizabeth here. The local reporting clause provides strong evidence that the model's speaker attribution is correct and the human label is wrong. The disagreement remains in the unadjudicated quantitative results; no post-hoc score correction is made.

This case demonstrates why even a near-ceiling speaker score should be called reference agreement rather than unquestioned accuracy.

### 5.2 An implicit speaker: the question about Miss Darcy's height

At worksheet row **1395**, the human reference attributes the reply about Miss Darcy's height to Mr. Bingley. V3, v5, and v6 attribute it to Darcy; v6 is paragraph **296**, turn **239**. The surrounding question concerns Darcy's sister, but the reply itself has no explicit speaker tag. Darcy is a plausible attribution; the human choice is not secured by an explicit reporting clause. This should be adjudicated, not automatically counted as a proven model mistake. The nearby addressee labels should be reviewed together with the speaker label.

### 5.3 Collective speech versus a single speaker

At worksheet row **10375**, the reference assigns the request to take the uncle's letter to `Jane + Elizabeth`. V6, paragraph **1644**, turn **1296**, assigns Elizabeth alone and lists Jane as a bystander. The surrounding narrative places both sisters in the interaction but does not explicitly establish simultaneous speech. A collective annotation and a single-speaker interpretation encode different assumptions. V6's choice is plausible, but not proven. This is an adjudication case rather than a reliable automatic correction.

### 5.4 A genuine unresolved attribution: the waiter is dismissed

At worksheet row **7532**, the human reference attributes the reported instruction dismissing the waiter to `Jane + Elizabeth`. V6, paragraph **1216**, turn **993**, uses `Unknown` as speaker. The passive construction establishes the instruction but not which sister delivered it. The model's abstention is defensible under a conservative identity policy, although it loses recall against the collective human label. A group-level speaker label and an unknown individual speaker should not be treated as semantically identical.

### 5.5 Anonymous local roles are not failed named-entity resolution

At worksheet row **10207**, v6 paragraph **1602**, turn **1264**, the human speaker is `the butler`. V6 retains `S="Unknown"` but adds `S_ref="the butler"` and `S_ref_p="1601"`. This preserves a local role without inventing a global character identity. The speaker is excluded from named-identity scoring by the stated policy. The model nevertheless differs on audience structure: it addresses Jane and Elizabeth, whereas the reference addresses Jane and treats Elizabeth as a bystander. Local-role preservation and addressee correctness are separate questions.

### 5.6 Splitting when the addressee changes

In the early family conversation, the workbook separates the father's general reply from his subsequent address to Mary, and later separates the mother's address to the daughters from her address to Lydia. V5 merges these shifts into broader turns: its early turn **46** combines the reply and address to Mary; turn **53** combines the group address and address to Lydia. V6 restores the splits at turns **46–47** and **54–55**. This is consistent with the later prompt's mandatory splitting when the actual addressee subset changes. It is a concrete boundary/addressee improvement, although a single example cannot quantify the contribution of the review stage.

### 5.7 Listener persistence and unsupported expansion

In Chapter II, v4 initially lists only Mrs. Bennet as listener to the father's address to Elizabeth. V6 turn **31** lists the mother and all four other daughters. Later reference turns establish a larger family audience, but the first human turn itself has a blank bystander cell. This exposes two risks: v4 can under-propagate scene participants, while v6 can back-project later-established presence into earlier turns. A blank cell cannot resolve which interpretation is correct. Physical co-presence, audibility, and active listening must be distinguished in adjudication.

### 5.8 Over-inclusive named listeners in the Pemberley conversation

At worksheet row **9196**, v6 paragraph **1454**, turn **1156**, the human reference lists Miss Darcy as bystander. V6 additionally lists Mrs. Annesley and Mrs. Hurst. This is a concrete listener-set disagreement and illustrates how a model can obtain the binary presence decision right while losing identity precision. The additional people should be checked against scene continuity and audibility, rather than accepted merely because they occur in the surrounding chapter. The present comparison does not independently prove their active attention.

### 5.9 Unmarked reported speech and thought inflate apparent false positives

Near the beginning of Chapter III, the human workbook leaves the favourable report by Lady Lucas and the family's evaluative reaction without complete marked turns. The models annotate portions as FID. Some represented report or perspective is plausibly present, so an unmatched model span is not automatically hallucinated dialogue. Conversely, labelling a report as FID rather than IS is an interpretive decision requiring explicit guidelines.

The early report of Bingley's inability to accept an invitation is labelled FID in the supplied outputs. Its written-message context makes communicated indirect report a plausible alternative. This is a useful subtype-adjudication candidate, not a basis for declaring a corpus-wide FID error rate.

### 5.10 Scope change: silent reported thought and self-address

Earlier outputs annotate Mrs. Bennet's inability to imagine Bingley's business and her subsequent fear as IS, with herself as addressee. V6 leaves these passages unannotated under the later policy excluding silent reported thought from IS. Elsewhere, v6 marks genuine FID with an empty addressee and an explicit thought/self-address attribute. These differences are not ordinary missed turns under an unchanged task definition. They show that the pipeline's ontology changed during development.

## 6. Version-by-version interpretation and script effects

### 6.1 XML v3, generated with script v7

V3 is the first evaluated baseline. It already has strong speaker and addressee agreement on matched turns, and the highest conditional addressee membership F1 among the four outputs. Its principal weaknesses are six nested turns, over-inclusive or uncertain audience reconstruction, and a broad treatment of represented content that is not fully matched by the workbook. Its conditional speaker strength should not obscure lower end-to-end performance.

### 6.2 XML v4, generated with script v10

The v10 script explicitly requires flat turns, increases validation attempts, introduces task-specific thinking settings, and uses preceding context plus older scene cues. The observed removal of nested turns is directly consistent with the structural rule. Listener identity and presence improve over v3, while turn detection and addressee agreement do not improve. V4 has the best positive-row indirect-flag coverage. The effects of thinking level, retries, and context cannot be separated without controlled reruns.

### 6.3 XML v5, generated with script v11

The v11 script adds following reference context, character-limited preceding/following windows, and a mandatory `NO`/`IS`/`FID` attribute. Its defaults disable older cue anchors. V5 produces 1,805 turns, the largest output, and the most IS spans. Nevertheless, it has the lowest relaxed and exact detection F1 and lower addressee agreement. The early addressee-shift merges demonstrate a specific boundary weakness. Following context therefore cannot be claimed to have improved overall accuracy merely because it was added.

### 6.4 XML v6, generated with script v16_1

The supplied file's header describes a v16 pipeline, and the output declares schema `16.0`; the filename identifies the supplied revision as v16_1. The later pipeline includes separate letters, explicit empty self-address, narrower IS scope, restored addressee-change splitting, older scene anchors with a separate budget, registry discovery, local anonymous-role provenance, and targeted priority review. The observed output has the strongest span and listener agreement, including on the common matched cohort.

These changes are bundled. The results do not show that resumable registry handling itself improves literary accuracy, that every local role is semantically correct, or that priority review caused the aggregate gains. The output alone does not reveal which individual turns were revised by review. Reliability engineering and semantic accuracy must remain distinct claims.

### 6.5 Causal interpretation

The comparison is longitudinal and observational, not a controlled ablation. There is one output per version; prompts, schema, scope, context, and review policy change together. No repeated-run variance, provider-version stability, or held-out-novel performance is measured. The appropriate wording is “consistent with”, “associated with”, or “observed in”, not “caused by”.

## 7. Limitations and threats to validity

1. **Reference incompleteness:** the 1,383-span benchmark excludes unmarked content and unresolved marker sequences. Apparent precision is not absolute literary precision.
2. **Reference correctness:** explicit attribution errors and debatable collective labels remain uncorrected in the primary scores.
3. **Metadata reconstruction:** ordered assignment is deterministic but can misassociate a label in malformed groups. Three detected ambiguous assignments are excluded; undetected ambiguity remains possible.
4. **Boundary normalisation:** punctuation and whitespace are ignored. Exact-normalised boundaries are not original-text exact boundaries. Edited-region boundaries use a nearest-aligned fallback.
5. **Scope non-equivalence:** separate letters, silent thought, self-address, and splitting policy change across versions. No fully harmonised spoken-only benchmark is claimed.
6. **Identity equivalence:** canonicalisation resolves many superficial differences but can conceal ambiguous aliases. Raw-name agreement is not a substitute for semantic evaluation.
7. **Missingness:** blank bystander cells are treated as absence and blank span-level indirect labels as negative. Both assumptions can overstate false positives.
8. **Active listening:** participant lists do not independently establish attention. The listener objective is evaluated through annotation agreement, not behavioural evidence.
9. **Subtype scarcity:** only one explicit FID row exists, with uncertain scope. Generic indirect labels cannot support IS/FID classification scores.
10. **Conditional selection:** high attribution scores apply to matched, label-resolved turns. Common-cohort and end-to-end sensitivities reduce, but do not eliminate, this limitation.
11. **No inferential independence:** turns within a scene are correlated. No confidence interval or significance claim is made from an inappropriate independent-turn model.
12. **Development-set reuse:** this novel has been used during iterative pipeline development. These are development-corpus results, not evidence of generalisation across the full Austen corpus.
13. **Single-run evidence:** there is no replication, runtime cost comparison, or controlled ablation. Self-reported XML confidence is not calibrated and is not used to weight correctness.
14. **Qualitative sampling:** the examples are diagnostic, not a prevalence study. Potential reference corrections require independent adjudication before changing the benchmark.

## 8. Conclusions and recommended next experiment

For the highest-priority objectives, **XML v6 is the strongest overall candidate**, principally because of its higher turn-span agreement and substantially better listener reconstruction. It does not dominate all measures: v3 retains the highest conditional addressee F1, and v4 covers the largest proportion of positive indirect-speech rows. Speaker identification is already close to the agreement ceiling on the matched subset, so further work should prioritise boundaries, recipient changes, audience continuity, and reference adjudication rather than interpreting tiny speaker-score differences.

The next evaluation should freeze an explicit annotation protocol and adjudicate a stratified sample containing: direct dialogue, recipient changes, multi-party scenes, overhearing, anonymous roles, letters read silently/aloud, communicated IS, silent reported thought, and FID. Two independent annotators should mark exact spans and participant roles, with disagreement resolution documented separately. Preserve both the original workbook and an adjudicated version. Report uncorrected and adjudicated results side by side.

Then rerun the same frozen model/configuration with controlled changes to following context, older anchors, registry discovery, and priority review. Use a different Austen novel as held-out material. For subtype evaluation, collect an explicit IS/FID-labelled sample before reporting macro-F1. For listener evaluation, define separately physical presence, plausible audibility, and active listening.

The central result is therefore a qualified one: the latest pipeline improves agreement on several operationally important layers, but a clean estimate of literary accuracy requires a more consistent reference and a stable annotation ontology.

## Appendix A. Marker anomalies

These are parser-detected events at worksheet rows, not adjudicated literary errors. Unclosed openings are relative to nonblank row groups; the original workbook remains authoritative for manual review.


| Worksheet row | Event |
| --- | --- |
| 667 | orphan close |
| 668 | orphan close |
| 1081 | unclosed |
| 1090 | orphan close |
| 1132 | unclosed |
| 1141 | orphan close |
| 3022 | double open |
| 3022 | unclosed |
| 4410 | orphan close |
| 7502 | double open |
| 7503 | double open |
| 9675 | double open |
| 9675 | unclosed |
| 12176 | unclosed |
| 12183 | orphan close |
| 12460 | unclosed |
| 13659 | double open |
| 13659 | unclosed |

## Appendix B. Reproducibility and source fingerprints

The accompanying `evaluate_austen.py` computes the comparison using Python, `openpyxl`, NumPy, SciPy, and RapidFuzz. It performs no model calls and does not execute the supplied annotation scripts. It reads their literal alias dictionary through Python's AST. `austen_evaluation_results.json` contains unrounded metrics, confusion counts, source hashes, and matched disagreement records. For repository use, place these two evaluation companions beside this report. By default, the evaluator reads inputs two directory levels above its own location and writes results beside itself. Optional `--data-dir` and `--output-dir` arguments override these paths. The evaluation performed for this report used the supplied files in `/mnt/data`. The source links in this report assume the two-level relative layout requested by the researcher.

All results derive from the supplied files. No external secondary account or prior-chat metric is substituted for this computation. The SHA-256 fingerprints below identify the exact inputs evaluated.


| Input | SHA-256 |
| --- | --- |
| `annotate_austen-v10.py` | `a5e8c9d7756d286a8fa4d7d5237cae36f12f667e9a06692166054d6593c0092f` |
| `annotate_austen-v11.py` | `37a65ac9e3df2088797a8a78bf9e75faed4bcae3ffcd7caa6d1c8bb9477d2277` |
| `annotate_austen-v16_1.py` | `0d82b64b1e83cd775671295e2932b252334c35f35948267473ae3a75d105da85` |
| `annotate_austen-v7.py` | `ce12aeaa8e0aced641b384630dbc87ce831ea3fe56d7d64676be02b27a6ceef5` |
| `pride_and_prejudice_JK.xlsx` | `c728a309573cb78b37930de6eed2aea75e3012ede9c1917fccd81dd3b3d0fa49` |
| `pride_and_prejudice_annotated_v3.xml` | `93c0e442150e7484c7a853fd9886ea4dfd5e605c5357d36fccefc5452a7371d4` |
| `pride_and_prejudice_annotated_v4.xml` | `adfd01974d3a43b0f28d7a176dd0567cf760b730b4faca1db84da95e1e7a75a2` |
| `pride_and_prejudice_annotated_v5.xml` | `62d1e065ab77866eaf7e66422cb4307f70786d20757c1a7bbbfdf05731d16921` |
| `pride_and_prejudice_annotated_v6.xml` | `2ff43edd5761a2e3394bef1b6457de72b428e1c2214505c3a1b9653aaa07be36` |
