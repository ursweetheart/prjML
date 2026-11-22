# Notes

## Corpus (task 5.1), 2026-11-22

- Source: Vietnamese Wikisource page "Truyện Kiều" (Nguyễn Du, quốc ngữ text), `https://vi.wikisource.org/wiki/Truyện_Kiều`, revision 206607 (page touched 2026-11-07). The page carries the `{{PD-old}}` public-domain tag; the author died in 1820.
- Cleaning: keep the text between `<poem>` and `</poem>`, delete the `{{số|N}}` line-number templates, NFC-normalize, strip each line, drop empty lines.
- Result: 3,254 lines (the canonical length), 104,204 characters, 132 distinct characters. The proposal's "about 100 characters" was an estimate; 132 includes uppercase letters, tone-marked vowels and the punctuation of the dialogue quotes. The A.4 sampling check is unaffected.
- The file is `data/raw/truyen_kieu.txt` (git-ignored). The recipe to rebuild it is in the README.

## Training run (task 5.1), 2026-11-22

Default constants, 3,000 steps each, GPU, 2 min 19 s for both models. Uniform-guess loss ln V = 4.883.

| model | parameters (layers / width / heads) | best validation loss | at step |
| --- | --- | --- | --- |
| target | 4 / 128 / 4 | 1.611 | 500 |
| draft | 2 / 64 / 2 | 1.589 | 2,750 |

Both beat ln V by a wide margin. Observations, not failures:

- The target overfits: training loss falls to 0.20 while validation loss rises from 1.611 (step 500) to 2.778 (step 3,000). The saved weights are the step-500 ones, so what is saved is an under-trained target.
- Because of that, the draft's validation loss is slightly lower than the target's. A.4 correctness does not need the target to be better, but the B.3/C.3 acceptance-rate experiments, and a realistic speedup story, are easier to explain with a target that is clearly stronger. Options if wanted: more dropout or weight decay for the target, or a smaller learning rate with evaluation every 100 steps. Not done here: the design leaves tuning out of this change.
- Samples of 300 characters keep the lục bát line structure and many real words ("Trăm năm", "nàng", "lòng", "tình"), with some non-words ("thỏt khenền"), which is expected for a character model on 100,000 characters.

## Target tuning (task 6.2), 2026-11-22

Target only, 3,000 steps, evaluation every 100 steps, draft untouched (saved draft validation loss 1.589, measured again with the same estimator after the runs). Goal: target at least 0.05 lower, i.e. 1.539 or better.

| run | dropout | weight decay | lr | best validation loss | at step | gap to draft |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (first run, before group 6) | 0.1 | 0.01 | 3e-3 | 1.611 | 500 | -0.022 |
| 1 | 0.3 | 0.1 | 3e-3 | **1.557** | about 1,200 | 0.032 |
| 2 | 0.4 | 0.1 | 3e-3 | 1.593 | | -0.004 |
| 3 | 0.3 | 0.1 | 1e-3 | 1.575 | about 2,400 | 0.014 |
| 4 | 0.3 | 0.3 | 3e-3 | 1.585 | | 0.004 |

The criterion was **not reached**; the cap of four runs was used. The settings of run 1 are kept (target dropout 0.3, weight decay 0.1, lr 3e-3) and its checkpoint is the saved `target.pt`: validation loss 1.557 against the draft's 1.589, a gap of 0.032 with 827,268 parameters against 117,124.

Why the gap stops there: the corpus is about 94,000 training characters, and the 2-layer draft already reaches 1.589, close to what this much text supports. At lr 3e-3 the target stops improving between step 500 and step 1,800 (at 1e-3, around step 2,400), so extra capacity has little left to learn. Differences between runs of 0.01 to 0.02 are within what a single run and 20 fixed validation windows can show, and no repeated seeds were run.

What this means for B.3/C.3: the target is better than the draft, but only slightly. Acceptance rate will be controlled mainly by how little the draft is trained (a draft at 250 steps has validation loss 1.96), not by a large gap between fully trained models. If a wider range is needed, options not tried here: a smaller draft (1 layer, width 32) or a target trained on more text.
