# Comparative annotation analysis of *Sense and Sensibility*
## A reference-free study of a fine-tuned Llama 3.3 70B output and two Gemini pipeline versions

## Abstract

This report compares three machine-annotated editions of *Sense and Sensibility* across seven prioritised objectives: turn detection, speaker identification, addressee identification, listener presence, listener identity, indirect-representation detection, and indirect-representation subtype. No human annotation of this novel is available. The earlier Llama output is therefore treated as a **comparison system, not a reference standard**. Quantitative evidence consists of structural audits, annotation distributions, symmetric span overlap, conditional participant agreement, and character-level coverage. Targeted close reading distinguishes clear local errors, plausible alternative readings, and changes in annotation policy.

The two Gemini outputs share a stable core of dialogue attribution, while the later output removes nested turns and introduces separate letters and explicit self-address. The Llama file contains substantially fewer annotated turns, explicit mapping failures, unstable identity expressions, and almost no positive indirect-speech labels. These observations support a practical preference for the later Gemini output as a research starting point, but not a numerical claim of superior corpus-wide accuracy. The later pipeline also omits some communicated content that remains within its stated scope. Reduced annotation volume must not automatically be interpreted as improved precision.

## 1. Materials and provenance

The source files and scripts are assumed to be located **two directory levels above this report**. Links use the supplied filenames, which contain a hyphen before the script version rather than the underscore used in the task description.


| System label | Annotation file | Generating pipeline |
| --- | --- | --- |
| Llama | [sense_and_sensibility_annotated_llama70B.xml](../../sense_and_sensibility_annotated_llama70B.xml) | Researcher-reported Llama 3.3 70B fine-tuned on human Pride and Prejudice annotations |
| v1 | [sense_and_sensibility_annotated_v1.xml](../../sense_and_sensibility_annotated_v1.xml) | [annotate_austen-v7.py](../../annotate_austen-v7.py) |
| v2 | [sense_and_sensibility_annotated_v2.xml](../../sense_and_sensibility_annotated_v2.xml) | [annotate_austen-v16_1.py](../../annotate_austen-v16_1.py) |

The Llama root records `Llama-3.3-70B-Instruct-LoRA`, a minimum-generated-token-probability confidence method, and a prompt regime described as identical to fine-tuning. Both Gemini roots record `gemini-4-argon`; v2 additionally declares schema `16.0`, separate letters, explicit empty self-address, and local anonymous references. These are file metadata, not independently verified model specifications. The researcher supplies the training provenance and script-to-output mapping. Training data, training logs, the Llama inference script, runtime command lines, and complete run checkpoints are not available for independent verification.

The supplied Python scripts are inspected as source, not executed as annotation pipelines. Their defaults and documented policies describe intended behaviour; they do not establish which optional settings were used in the historical runs. No external literary interpretation is substituted for the supplied source text.

## 2. Research design

### 2.1 What can and cannot be measured

Without an independently annotated reference, true positives, false positives, false negatives, accuracy, precision, and recall **cannot be estimated as correctness metrics**. Calling Llama the gold standard would merely reward reproduction of its errors and annotation conventions. Likewise, agreement between the two Gemini runs is not independent confirmation: they share a model identifier, pipeline ancestry, and substantial prompt logic.

This report uses symmetric agreement statistics instead. A Dice coefficient is algebraically related to F1, but here it measures overlap between two machine outputs, not performance against truth. Directional coverage is labelled coverage, never recall. Local source-based judgements are explicitly qualitative and are not extrapolated into a corpus-wide error rate.

### 2.2 Objectives


| Objective | Priority | Available evidence |
| --- | --- | --- |
| 1. Turn detection | Must | Counts, nesting, mapping failures, pairwise span and character overlap |
| 2. Speaker identification | Must | Canonical set agreement, identity-resolution coverage, close reading |
| 3. Addressee identification | Must | Set agreement, unresolved expressions, self-address and group recipients |
| 4. Listener presence | Must | Positive rates, matched-turn contingency tables, Cohen kappa |
| 5. Active listener identity | Should | v1–v2 named-set agreement only; Llama has binary labels |
| 6. Indirect representation present | Should | Label prevalence, binary agreement, positive character overlap |
| 7. Precise indirect type | Could | v1–v2 subtype cross-tabulation and qualitative adjudication candidates |

### 2.3 Full-file parsing and alignment

All three XML files are parsed in full. Each has **1,923 paragraph elements**. For every corresponding paragraph, lowercase alphanumeric text reconstructed from XML is identical across all three files. This supports direct paragraph-local alignment without fuzzy matching. It does not establish byte-level identity: punctuation, whitespace, and underscores are removed for this check.

Offsets count normalised alphanumeric characters within a paragraph. Every `<turn>` is retained, including nested turns; `<letter>` elements are counted separately and excluded from primary turn matching. The zero-based paragraph index used throughout this report includes front matter and is not a chapter-relative index. Turn identifiers use `number` in Llama and `n` in Gemini, so identifiers are not assumed to align.

Within each paragraph, candidate spans are matched one-to-one at intersection-over-union **IoU ≥ 0.50**. Assignment maximises the number of admissible matches first, then their total IoU. Thus a split or merge can yield only one matched pair per span. Exact agreement means equal start and end positions in normalised coordinates, not punctuation-exact boundaries. Pairwise span Dice is `2M / (N_left + N_right)`, where M is the number of matches. Directional coverage is `M / N_system`.

A supplementary character Dice uses the union of annotated character positions in each system, counting overlapping nested spans only once. It is less sensitive to splitting, but long turns dominate it. The indirect-only character measure uses only positively labelled IS/FID/YES spans.

### 2.4 Label harmonisation and identity limitations

`no`, `NO`, and an absent indirect-speech attribute are treated as the non-indirect class. Llama `yes` is positive indirect representation but **not** converted to IS or FID. Gemini `IS` and `FID` are positive. For listener presence, Llama `yes` is positive; a nonempty named Gemini listener list is positive; `no` is negative. This harmonises presence only, not listener identity.

Participant expressions are lowercased, split on `+` or the word `and`, and compared as unordered sets. A fixed explicit-name alias dictionary is extracted from the supplied v7 script and applied to every output. `Mrs. John Dashwood` is reconciled with Fanny; explicit first names and full names are reconciled where the dictionary provides them. Context-dependent `Mrs. Dashwood`, `Mr. Dashwood`, pronouns, relational descriptions, and collective expressions are **not automatically resolved**. In particular, assigning every occurrence of `Mrs. Dashwood` to the mother would misread the early John–Fanny conversation.

Two identity views are reported: conservative canonical agreement over all matched turns, and a restricted subset where both sides contain only identities recognised by the fixed dictionary. The latter is a selected easier subset, not an unbiased estimate of attribution accuracy. The dictionary is incomplete for minor characters and title variants; `Dr. Harris` versus `Mr. Harris`, for example, can lower agreement without indicating a different person. No context-sensitive entity resolver or manual global relabelling is silently applied.

### 2.5 Qualitative sampling

Examples are selected purposively from explicit attribution conflicts, missing dialogue, listener discontinuities, scope changes, anonymous roles, and subtype disagreements. They cover early dialogue and later service/illness scenes, but are not a representative random sample. Findings are classified as clear local problems, plausible readings, or policy-dependent differences. Agreement with another model is never the sole reason for declaring an annotation correct.

## 3. Structural audit and annotation volume


| System | Turns | Separate letters | Nested turns | Mapping-failed paragraphs | Unannotated quote-containing paragraphs |
| --- | --- | --- | --- | --- | --- |
| Llama | 1002 | 0 | 0 | 101 | 138 |
| v1 | 1588 | 0 | 55 | 0 | 9 |
| v2 | 1311 | 20 | 0 | 0 | 10 |

The quote diagnostic counts paragraphs containing an opening curly quotation mark but neither a turn nor a letter. It is **not a count of missed spoken turns**: quotation can represent thought, a cited phrase, or a different annotation scope. It nevertheless provides a useful screening signal, especially when combined with explicit `mapping_failed` status and close reading.

The 101 Llama mapping failures are output-level failures, not proof that the underlying language model failed to recognise speech. Without the inference/mapping code, recognition errors and export/alignment failures cannot be disentangled. They still matter for the usability of the delivered corpus.

V1 has 55 nested turns, incompatible with the later flat-turn protocol. V2 has none. Llama is also flat, so flatness alone cannot establish adequate detection. V2 additionally contains 20 separate letters. These are not simply 20 conversational turns to be added back into a harmonised score: medium, reading aloud, and segmentation require separate definitions.

### 3.1 Representation labels


| System | NO / no | IS | FID | Generic YES | Positive indirect total |
| --- | --- | --- | --- | --- | --- |
| Llama | 999 | 0 | 0 | 3 | 3 |
| v1 | 1121 | 387 | 80 | 0 | 467 |
| v2 | 1083 | 172 | 56 | 0 | 228 |

The reduction from 1,588 turns in v1 to 1,311 in v2 is accompanied by a reduction from 467 to 228 positive indirect spans. This is partly compatible with excluding silent reported thought and changing letter handling, but not all removed spans fall outside communicated speech. Raw counts cannot establish precision or recall.

## 4. Quantitative comparison by objective

### 4.1 Objective 1: turn detection (Must)


| Pair, left → right | Matched spans | Left coverage (%) | Right coverage (%) | Span Dice (%) | Exact matched boundaries | Character Dice (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Llama → v1 | 967 | 96.51 | 60.89 | 74.67 | 925 | 85.00 |
| Llama → v2 | 953 | 95.11 | 72.69 | 82.40 | 907 | 86.36 |
| v1 → v2 | 1282 | 80.73 | 97.79 | 88.44 | 1247 | 95.02 |

Most Llama turns overlap a Gemini turn, but many Gemini turns have no Llama counterpart. This asymmetry reflects Llama's smaller annotation inventory, not measured Llama precision. V2 is almost a subset of v1 at the relaxed span level: 1,282 of its 1,311 turns match v1, while only 80.73% of v1 turns match v2. Character Dice is higher than span Dice, showing that segmentation and coverage differences are distinct issues.

The 55 nested v1 turns remain in the primary inventory because they are actually present in the file. Removing them would change the comparison; no hidden flattening is performed. The strict flat-turn protocol is therefore evaluated structurally rather than retroactively imposed on v1.

### 4.2 Objective 2: speaker identification (Must)


| Pair | Matched turns | Canonical exact-set agreement (%) | Membership Dice (%) | Both-resolved subset n | Exact agreement in subset (%) |
| --- | --- | --- | --- | --- | --- |
| Llama / v1 | 967 | 73.94 | 73.94 | 789 | 90.24 |
| Llama / v2 | 953 | 73.87 | 73.68 | 778 | 90.10 |
| v1 / v2 | 1282 | 98.99 | 98.85 | 1257 | 99.84 |

The Gemini pair has very stable speaker assignments on matched content. Of its 13 conservative speaker-set disagreements, seven concern the repeated `Dr. Harris`/`Mr. Harris` title difference. This is a naming issue rather than seven independently established speaker mistakes. Other differences include collective groups and conservative `Unknown` assignments.

Llama–Gemini agreement rises markedly when unresolved identity expressions are excluded. That does not mean the excluded expressions are wrong: a pronoun can be locally intelligible. It means they do not provide stable corpus-wide character identifiers. Conversely, the restricted agreement remains below 100%, and explicit local attribution errors exist.

### 4.3 Objective 3: addressee identification (Must)


| Pair | Canonical exact-set agreement (%) | Membership Dice (%) | Both-resolved subset n | Exact agreement in subset (%) |
| --- | --- | --- | --- | --- |
| Llama / v1 | 68.05 | 67.30 | 770 | 84.81 |
| Llama / v2 | 69.15 | 68.01 | 755 | 86.62 |
| v1 / v2 | 92.43 | 95.94 | 1222 | 95.99 |

| System | Speaker sets recognised by dictionary / turns | Addressee sets recognised / turns | Explicit Unknown speakers | Empty addressees | Explicit self-address turns |
| --- | --- | --- | --- | --- | --- |
| Llama | 820 / 1002 | 801 / 1002 | 37 | 0 | 0 |
| v1 | 1549 / 1588 | 1564 / 1588 | 15 | 0 | 0 |
| v2 | 1286 / 1311 | 1247 / 1311 | 7 | 42 | 42 |

V2 has 42 empty addressees and 42 explicit self-address turns, whereas v1 has no empty addressees. This is a protocol change, not evidence of 42 failed recipient assignments. Llama expressions such as `himself`, `his wife`, and `previous speaker` likewise require contextual interpretation before comparison with named or empty recipients.

Addressee membership Dice can exceed exact-set agreement because partially overlapping recipient groups receive partial credit. The two statistics should be retained together: the former can conceal one unsupported extra recipient, while the latter treats a one-person difference as a complete set mismatch.

### 4.4 Objective 4: listener presence (Must)


| System | Positive listener turns | Share of all turns (%) |
| --- | --- | --- |
| Llama | 355 | 35.43 |
| v1 | 554 | 34.89 |
| v2 | 504 | 38.44 |

| Pair | Neither positive (00) | Left only (10) | Right only (01) | Both positive (11) | Agreement (%) | Cohen kappa |
| --- | --- | --- | --- | --- | --- | --- |
| Llama / v1 | 512 | 100 | 113 | 242 | 77.97 | 0.522 |
| Llama / v2 | 510 | 93 | 103 | 247 | 79.43 | 0.555 |
| v1 / v2 | 765 | 20 | 25 | 472 | 96.49 | 0.926 |

These contingency tables concern matched turns only. Overall positive rates use different turn inventories and cannot be directly interpreted as sensitivity. The Gemini pair agrees strongly on listener presence, but local disagreements and shared over-inference remain possible. Llama's binary field supports this objective even though it cannot support named-listener evaluation.

### 4.5 Objective 5: identities of active listeners (Should)

Llama contains only `yes`/`no` bystander values. Its named-listener performance is therefore **not represented in this output**, rather than measured as zero accuracy. The absence of identities prevents a three-way named-listener comparison.


| Pair | Matched turns | Exact listener-set agreement (%) | Membership Dice (%) | Mean set Jaccard (%) |
| --- | --- | --- | --- | --- |
| v1 / v2 | 1282 | 89.24 | 92.74 | 94.26 |

Empty–empty listener sets count as exact agreement and have Jaccard 1; membership Dice pools positive identity memberships and is not increased by empty–empty cases. These measures capture different aspects of agreement.

Crucially, named presence does not prove **active listening**. Scene membership, physical proximity, plausible audibility, and attention are separate constructs. The files contain inferred listener lists, not independent evidence of attention. The defensible result is agreement on annotated listener identities, not validated active-listener detection.

### 4.6 Objective 6: indirect representation present (Should)


| Pair | 00 | 10 | 01 | 11 | Binary agreement (%) | Cohen kappa | Indirect-character Dice (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama / v1 | 929 | 2 | 35 | 1 | 96.17 | 0.046 | 0.15 |
| Llama / v2 | 919 | 0 | 34 | 0 | 96.43 | 0.000 | 0.00 |
| v1 / v2 | 1073 | 0 | 0 | 209 | 100.00 | 1.000 | 67.20 |

This table exposes two important selection effects.

First, Llama–Gemini binary agreement exceeds 96%, despite Llama having only three positive indirect labels. Agreement is dominated by the shared negative class and by matching mostly direct dialogue. Near-zero kappa and negligible positive-character overlap show why the headline percentage is misleading.

Second, v1 and v2 have **100% binary agreement on their 1,282 matched turns**, but indirect-character Dice is only 67.20%. Their matched positive spans agree about being indirect, while many additional v1 indirect spans are unmatched. Conditional classification agreement must therefore not be presented as complete indirect-speech detection agreement.

The near absence of positive Llama labels is not evidence that the novel lacks indirect representation. Source-based cases below show reported content annotated as `no` or left unmarked. The reason could involve training-label conventions, generalisation, inference, or post-processing; the supplied files do not identify which mechanism dominates.

### 4.7 Objective 7: precise indirect type (Could)

Llama exposes only a generic binary label, so IS-versus-FID agreement with it is not identifiable. No conversion of its `yes` to IS is justified.

For the Gemini pair, the full matched-turn cross-tabulation is:


| v1 type | v2 type | Matched turns |
| --- | --- | --- |
| IS | IS | 157 |
| FID | FID | 44 |
| NO | NO | 1073 |
| IS | FID | 8 |

There are 209 matched positive indirect spans: 201 retain their subtype and eight change from IS to FID. Conditional subtype agreement on these positives is **96.17% (201/209)**. Including 1,073 shared NO turns would inflate the apparent subtype agreement and is not used as the substantive result.

This is stability, not subtype accuracy. Shared errors remain possible, and the unmatched v1 positive spans are excluded from this denominator. Distinguishing communicated free indirect report, free indirect thought, ordinary IS, and narratorial summary requires explicit literary criteria and independent annotation.

## 5. Source-based qualitative comparison

The following cases use zero-based paragraph indices. File links in Section 1 and the accompanying turn-record JSON provide the evidence trail. Judgements concern these passages only; no representative error-rate estimate is claimed.

### 5.1 Clear Llama speaker error in the Devonshire announcement

**Paragraph 103:** the narrative explicitly identifies Edward as turning towards Mrs. Dashwood and repeating the question about Devonshire. Llama turn **50** instead assigns `Mrs. Dashwood` as speaker and `Mr. John Dashwood` as addressee. V1 turn **69** and v2 turn **62** identify Edward as speaker and Mrs. Henry Dashwood as recipient. The reporting clause supports the Gemini attribution. This is a concrete speaker error, not merely a difference in name spelling.

The same paragraph also shows why boundaries and types must be analysed separately: Llama combines the mother's explanation with the geographical statement and labels it non-indirect; v1 separates an IS summary from a FID-labelled geographical statement; v2 retains the latter as FID. The FID classification is debatable even where the speaker is plausible.

### 5.2 An unambiguous mapping failure versus a detection omission

**Paragraph 41:** the Llama paragraph is explicitly marked `mapping_failed` and contains no turn, although it contains a directly quoted reply in the John–Fanny exchange. V1 turn **18** and v2 turn **16** annotate the reply. This is a delivered-output omission with a recorded mapping problem.

**Paragraph 38:** Llama also leaves a directly quoted reply about reducing the proposed sum unannotated, but without the same explicit failure marker. V1 turn **15** and v2 turn **13** annotate it. The contrast shows that not every missing turn has an exposed mapping-failure explanation. Neither case permits attributing the whole problem solely to the language model rather than the pipeline.

### 5.3 Identity drift inside a stable two-person exchange

**Paragraphs 43, 46, and 53:** Llama uses `lady`/`he`, a malformed relational description of Fanny, and `Mr. Dashwood`/`his wife`. The two Gemini outputs maintain John and Fanny as stable identities. The local discourse can sometimes resolve the Llama expressions, but a corpus query cannot safely treat them as global character identifiers without another coreference stage.

The malformed relational description in paragraph 46 is not an acceptable substitute for a canonical identity. Automatically treating all bare `Mrs. Dashwood` occurrences as the mother would create additional errors in this same exchange. Conservative unresolved-name handling is preferable to superficially higher agreement obtained through an unsafe alias rule.

### 5.4 A second clear Llama attribution error: the reply expressing contempt

**Paragraph 99:** the mother responds to Fanny's insinuations and leaves the room. Llama turn **49** assigns the response to `Mrs. Ferrars`, a person mentioned in the preceding report rather than the responding participant. V1 turn **58** assigns Mrs. Henry Dashwood to Mrs. Fanny Dashwood, which fits the local narrative. V2 does not annotate that response summary, while retaining Fanny's earlier communicated content as turn **54**.

This case is deliberately mixed: v1 offers the more plausible participant attribution for the response, Llama has a clear attribution problem, and v2 avoids the attribution only by omitting the span. Whether a summary without explicit propositional content should count as a turn is protocol-dependent. Omission must not automatically be rewarded as correction.

### 5.5 Nested represented speech inside current dialogue

**Paragraph 149:** v1 places a represented earlier remark by Elinor inside Mrs. Dashwood's current turn, creating a turn within a turn. The later protocol keeps quoted/reported material inside the current speaker's flat turn rather than creating a nested conversational event. V2 therefore improves structural compliance and avoids confusing the current conversation with an earlier reported exchange.

This is not a claim that embedded speech is linguistically unreal. It is a distinction between an embedding-capable discourse representation and the project's required flat-turn schema. A separate embedded-report layer would be needed to retain both levels without violating that schema.

### 5.6 Self-address and apostrophe to a place

**Paragraph 110:** Marianne's solitary address to Norland is missing from the Llama output, whose paragraph carries a mapping-failure marker. V1 turn **82** uses Marianne herself as addressee. V2 turn **69** uses an empty addressee and `self_address="speech"`.

V2 explicitly distinguishes the lack of an interpersonal recipient from unknown identity. The passage also addresses a place rhetorically, so an ontology permitting nonhuman addressees might choose another representation. Under the supplied character-oriented protocol, empty self-address is coherent and should not be scored as missing recipient information.

### 5.7 Listener continuity around an explicit exit

**Paragraphs 145–147:** before Elinor leaves, Llama turn **64** marks no bystanders for her exchange with Marianne, while both Gemini outputs retain the mother. Paragraph 146 explicitly records Elinor leaving. In paragraph 147, Llama turns **66–67** mark bystanders as present, while both Gemini outputs mark none for the mother–Marianne exchange.

The explicit exit and the preceding three-person interaction support the Gemini continuity decisions more strongly than Llama's binary sequence. The evidence does not identify a named hidden listener for Llama's positive flag after the exit. This illustrates a scene-state problem that a single-turn presence percentage cannot diagnose.

### 5.8 An extra sister in the recipient set

**Paragraph 127:** Llama turn **55** addresses Marianne and Elinor; v1 turn **98** additionally includes Margaret; v2 turn **73** returns to Elinor and Marianne. The following narrative focuses on Marianne's reaction for Elinor's sake, supporting those two recipients, but does not by itself prove that Margaret could not hear the remark.

V2's narrower recipient set is plausible, not an independently established correction. The case separates intended recipients from everyone who may be present. A family member should not be added solely because a collective family label occurs nearby.

### 5.9 Conservative Unknown versus unsupported collective certainty

**Paragraph 163:** the passive report that Sir John was eagerly asked about Willoughby does not specify which member of the family asked. V1 turn **135** assigns a collective speaker; v2 turn **99** uses `Unknown`. Llama supplies no turn. V2's abstention is defensible because the source establishes an utterance without establishing its individual speaker. The collective v1 label is possible, but stronger than the explicit evidence.

**Paragraph 259:** v1 attributes the reported praise of the grounds to Sir John, whereas v2 uses `Unknown` for the passive declaration. Sir John is described as particularly enthusiastic nearby, so v1's inference is plausible. This is a weaker case for preferring Unknown: the distinction is between cautious and contextual inference, not a demonstrable factual correction.

### 5.10 Anonymous service roles retained without inventing names

**Paragraph 860:** v1 turn **750** and v2 turn **655** both use an unknown speaker for the reply about letters. V2 adds `S_ref="the footman"` and provenance paragraph **859**. **Paragraph 1441:** v2 turn **1036** similarly preserves `the servant` as a local role for the carriage announcement. Llama leaves both paragraphs without turns.

The local references improve inspectability without falsely merging unnamed servants into a global character. Literal source provenance supports traceability, not automatic semantic correctness. Across v2, five turns have a speaker or addressee role-reference attribute; this is not five newly identified named characters.

### 5.11 Scope reduction that is justified: silent reported thought

**Paragraph 21:** v1 separately annotates John's internal plan to increase the sisters' fortunes as IS with himself as addressee. V2 excludes that silent reported thought while retaining the subsequent FID passage, with empty addressee and an explicit thought marker. Llama annotates the FID-like passage but labels it `no` and uses `himself` as recipient.

The v1–v2 difference is consistent with the later rule that IS must be communicated speech or writing. The Llama label is a subtype/detection concern, not evidence that the passage is ordinary interpersonal direct speech merely because quotation marks are present. A stable protocol must separate representation form from audibility.

### 5.12 Scope reduction that may lose communicated content

**Paragraph 18:** the dying father's recommendation to his son is annotated as IS in v1 and omitted in v2 and Llama. The passage reports a communicated recommendation, not merely silent thought. Whether its object is sufficiently explicit propositional content is debatable, but excluding silent thought does not alone explain its removal.

**Paragraph 121:** v1 turn **94** annotates the family's promise to dine at the park; v2 and Llama omit it. A promise is a communicated act and its content is stated. Under a protocol covering communicated indirect content, this is a strong omission candidate for v2, even if the exact collective speaker set remains uncertain.

These cases prevent a one-sided interpretation of v2's smaller indirect inventory as purely beneficial pruning. V1 should be retained as a source of candidate omissions for review.

### 5.13 A shared subtype ambiguity: geographical explanation

**Paragraph 103:** both Gemini versions label the geographical statement about the cottage as FID, following the mother's explanation. It may instead be ordinary indirect communicated content or narratorial supplementation. Llama labels its broader explanation span `no`. No system supplies a secure adjudication merely by agreeing with another.

The practical requirement is to define whether FID includes free indirect *speech*, free indirect *thought*, or both, and how an unintroduced explanatory sentence differs from narratorial continuation. Without that definition, subtype stability is not evidence of theoretical validity.

### 5.14 A later naming difference rather than a new speaker

**Paragraph 1550 and subsequent illness reports:** v1 uses `Dr. Harris`, v2 `Mr. Harris`. The source explicitly names Mr. Harris in the same paragraph. Seven matched speaker disagreements arise from this repeated title difference. V2 is more source-faithful in title, but the outputs refer to the same local participant. Treating these as seven unrelated identity errors would exaggerate semantic disagreement.

### 5.15 Collective labels can still overreach in v2

**Paragraphs 313 and 315:** v2 expands collective complaint/question speakers to include `Careys`, whereas v1 has a narrower or unresolved set. The source uses broad collective expressions rather than naming each speaker. Such expressions establish group participation without proving that every listed person actually speaks. The additional family label also introduces a group identity alongside individual identities.

This remains an adjudication candidate, not an automatic v2 improvement. Registry expansion can preserve a source group while still over-specifying its participation in a particular utterance. Named-entity discovery and correct role attribution are separate tasks.

## 6. Interpreting script changes

The following mechanisms are grounded in the supplied scripts and observable output, but do not constitute controlled causal estimates.


| Change in later pipeline | Observed evidence | Permissible inference |
| --- | --- | --- |
| Flat-turn validation | 55 nested v1 turns; zero in v2 | Structural compliance improved |
| Separate letter layer | 20 v2 letter elements | Turn inventories are not scope-equivalent |
| Empty self-address plus explicit mode | 42 v2 empty addressees/self-address turns | A representational convention changed |
| IS limited to communicated speech/writing | Silent thought removed in paragraph 21 | Some reductions are policy-consistent |
| Following context and older scene cues | Documented in later script; listener lists often stable | Possible support for scene continuity, not isolated causal proof |
| Priority review and recipient-change splitting | Documented later safeguards | No per-turn review audit supplied to identify actual revisions |
| Registry discovery and local anonymous roles | Local provenance in paragraphs 860 and 1441 | Better traceability, not guaranteed attribution correctness |
| Resumable registry handling | Reliability mechanism in later source | No direct literary-accuracy effect established |

The later script documents following context of up to 30 paragraphs/6,000 source characters, older scene cues with a separate budget, and a priority-review limit of four paragraphs by default. These are defaults, not verified historical command settings. Its source also explicitly states that IS is communicated speech/writing rather than silent reported thought, and that genuine FID and directly quoted thought remain in scope. The supplied v7 source lacks these later controls.

No conclusion can isolate the effects of context, thinking settings, registry discovery, or review because the versions change several components simultaneously. The Llama comparison is even more confounded: model family, fine-tuning, prompting, inference, mapping, confidence generation, and annotation schema differ. This is a comparison of **delivered annotation systems**, not a clean benchmark of underlying model capability.

## 7. Confidence and internal consistency


| System | Median turn confidence | Speaker/addressee confidence available | Review-recommended turns |
| --- | --- | --- | --- |
| Llama | 0.6816 | Yes | 291 |
| v1 | 0.9800 | No separate fields | 0 |
| v2 | 0.9800 | No separate fields | 0 |

Llama's confidence is recorded as a minimum generated-token probability; Gemini confidence is self-reported. These quantities are not commensurable probabilities of literary correctness. Llama's high speaker-token confidence can coexist with a source-contradicted speaker assignment, as in the Devonshire case. No calibration, Brier score, or confidence-based accuracy curve is possible without adjudicated labels.

No system has a detected speaker–bystander or addressee–bystander set intersection under the conservative normalisation. This is a useful internal consistency observation, not evidence that the listener sets are correct. Speaker–addressee overlap occurs in 20 Llama, 114 v1, and two v2 turns; self-address and collective conversation can legitimately produce such overlap, so these are not automatically errors.

## 8. Limitations and threats to validity

1. **No human reference:** all numerical results are agreement or descriptive measures. Local close reading does not create a full gold standard.
2. **Correlated systems:** v1 and v2 share model/pipeline ancestry; consensus can reproduce a shared mistake. Llama's reported training on a human annotation of another novel does not make its output authoritative here.
3. **Unequal task representation:** Llama lacks named listeners and subtype labels; v2 separates letters and changes self-address and silent-thought scope.
4. **Conditional selection:** participant and label agreement excludes unmatched spans. This particularly inflates indirect-label agreement on the shared direct-dialogue core.
5. **Segmentation:** one-to-one IoU matching penalises splits/merges; character Dice mitigates this but weights long spans heavily. Nested v1 turns complicate event counting.
6. **Text normalisation:** punctuation and whitespace are ignored. Exact normalised boundaries are not exact original-text boundaries.
7. **Conservative aliases:** pronouns, relational labels, bare titles, and groups remain unresolved. The restricted dictionary-resolved subset is easier and not representative of all turns.
8. **Attention is unobserved:** named presence does not establish active listening. Audibility and attention need independent operational definitions.
9. **Pipeline failures:** Llama mapping failures are observable, but their underlying causes cannot be diagnosed without inference code and logs.
10. **Qualitative selection:** diagnostic cases are purposive and not suitable for estimating prevalence or confidence intervals.
11. **Single runs and bundled changes:** there is no repeated-run variance, ablation, cost comparison, or verified historical runtime configuration.
12. **Training and evaluation independence:** the researcher reports training on Pride and Prejudice rather than this novel, but the complete training/pretraining provenance is not audited. No claim of uncontaminated generalisation is made.
13. **No causal or statistical significance claim:** correlated turns and scenes are not treated as independent observations. Kappa describes agreement, not correctness or a causal effect.

## 9. Conclusions and recommended research workflow

### 9.1 Findings by priority

For **Must** objectives, v2 is the most usable starting point among these delivered files: it is flat, uses more stable character identities than the Llama output, and supports explicit recipient and listener analysis. The qualitative audit identifies clear Llama attribution and output-omission problems. However, v2 is not demonstrated to have the best true turn recall, and its conservative omissions require review.

For the **Should** objectives, named listeners can be compared only between v1 and v2. Their strong agreement does not validate active attention. Indirect-representation coverage is substantially broader in v1; v2 removes both policy-incompatible thought annotations and some plausible communicated content. Llama's three positive labels and binary-only bystanders make it unsuitable as the sole benchmark for these layers.

For the **Could** objective, v1–v2 subtype stability can be measured, but IS/FID correctness remains unresolved. Llama cannot supply a subtype reference.

### 9.2 Recommended use of the three outputs

Use **v2 as the working annotation**, retain **v1 as an omission-discovery and alternative-boundary source**, and retain **Llama as an independent-system disagreement signal**, not a gold standard. Do not automatically take a majority vote: two Gemini outputs are correlated, and an unsupported Llama label should not outweigh an explicit reporting clause.

Create a small adjudicated evaluation set stratified by direct dialogue, recipient changes, multi-party scenes, exits, anonymous service roles, letters, communicated IS, silent thought, and FID. Two independent readers should annotate exact spans and roles under a frozen protocol, with disagreements adjudicated separately. Include both shared and unmatched spans to avoid evaluating only easy consensus cases. Only then report precision/recall as correctness metrics.

A particularly efficient next audit is to review the **v1-only indirect spans** and the **Llama mapping-failed paragraphs**. The former distinguish useful pruning from lost communicated content; the latter distinguish corpus-export failures from model interpretation. Freeze model/configuration and vary one pipeline feature at a time before attributing improvements to context or review.

The central conclusion is methodological as well as practical: v2 improves structural and representational discipline, but neither fewer annotations nor higher inter-model agreement proves greater literary accuracy. A source-grounded, protocol-stable adjudication layer is the necessary next step.

## Appendix A. Reproducibility

The companion [comparison script](compare_sense_and_sensibility.py) parses the supplied files, computes all descriptive and agreement metrics, and exports [full comparison results](sense_and_sensibility_comparison.json) and [turn records](sense_and_sensibility_turn_records.json). It requires Python, NumPy, and SciPy. It performs no model calls and does not execute the annotation scripts. Defaults read inputs two directory levels above the evaluator and write results beside it; `--data-dir` and `--output-dir` override those paths.

The comparison JSON contains unrounded metrics, per-chapter counts, source fingerprints, and matched disagreements. The turn records contain paragraph indices, offsets, original attributes, and text for every turn. Counts are independently checked against XML inventories, and contingency-table totals are checked against matched-pair counts. All displayed percentages are rounded to two decimals.

### Input SHA-256 fingerprints


| Input | SHA-256 |
| --- | --- |
| `sense_and_sensibility_annotated_llama70B.xml` | `d4b0c2bb6f9742f1e993d58163722002b176aa96505b55db5907cc720df9363b` |
| `sense_and_sensibility_annotated_v1.xml` | `51364b9b639a1879013731f7d19dc2f21b0a25aced3fde62b3a1c478b1ff73ff` |
| `sense_and_sensibility_annotated_v2.xml` | `3d48bf4129a75f2a42d41c10e6cffe45d9443cc72caf26b3e41703e79d617f4c` |
| `annotate_austen-v7.py` | `ce12aeaa8e0aced641b384630dbc87ce831ea3fe56d7d64676be02b27a6ceef5` |
| `annotate_austen-v16_1.py` | `0d82b64b1e83cd775671295e2932b252334c35f35948267473ae3a75d105da85` |
