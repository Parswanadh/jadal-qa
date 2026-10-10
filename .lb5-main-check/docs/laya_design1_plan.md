# Design 1 plan: VisionFusionDecisionModel (no upstream edits)

Line numbers refer to `/home/parshu/projects/awing/prep/laya_src/`. Its imports (`.agent`, `.backends`, `.fast`) do not match these file names, so confirm the package layout first.

## 0. Key findings

- **`b` is redundant.** Softmax and the act features are shift-invariant, so a constant added to all options has zero gradient under soft-CE. Drop it.
- **`w` and `s` are redundant.** Only `w*s` matters. Learn one scalar per type, `a = zeros(3)`, with a fixed scale `s`.
- **Image capability is unverified.** Nothing in the source mentions EmbeddingGemma or images.
- **Order test needs parallel layout** (`parallel_layout` docstring, laya_common.py:279).

## 1. Injection point

- `DecisionModel.forward` is at laya_common.py:609-659. Logits come from `self.scorer(m)` at line 633 and are masked at line 634. Shape is `(B, kmax)`.
- Inject between lines 633 and 634: `logits = logits + a[qtype][:, None] * option_sim`, with `option_sim` zero-padded `(B, kmax)`. Then the existing `masked_fill(~marker_mask, -1e4)` keeps padding masked.
- Do not call `super().forward` and add the term afterwards. Line 636 (entropy) and line 652 (`act_input`) would see unadjusted logits. Copy lines 609-659 into the subclass and let T1 guard the copy.
- Type index: `QTYPES = {"choice": 0, "score": 1, "noul": 2}` (laya_common.py:18), matching `type_emb = nn.Embedding(3, d)`.
- `a = zeros(3)` starts the model exactly at the unmodified Laya logits.

## 2. Option order

- `build_head` (laya_common.py:225) emits one `[MASK]` per option in `order`. Slot `s` holds canonical option `order[s]`, and `markers[s]` is its position. Canonical options come from `render_options` (laya_common.py:127). `noul` always has 2.
- `encode_item` (laya_train.py:446-462) permutes the target at lines 456-457: `target = [target[i] for i in option_order]`. Do the same for similarity: `option_sim = [sim_canon[i] for i in option_order]`.
- `draw_option_order` (laya_train.py:464) permutes only types in `shuffle_options`, which defaults to `()` (line 105). Training is canonical unless this is set.
- Inference: `option_order` comes from `_to_internal` (laya_agent.py:1128-1129), is used in `_encode_state` (line 1158), and is undone by `unpermute_probs` (laya_agent.py:1350; defined at laya_common.py:557). The similarity is slot-space and needs no unpermute, but the wrapper must permute `option_sim` with `q["option_order"]`.
- **Main risk:** similarity permuted by a different rule than the markers or target. Compute `option_sim` inside `encode_item` from the same `option_order` variable, and test it (T2).

## 3. Batching, training, loss

- `collate_items` (laya_common.py:893) builds `(n, kmax)` tensors from `markers` and `target`. Other keys fall into `meta`. A vision collate wrapper should call `collate_items`, pop `option_sim` into a zero-padded `(n, kmax)` float tensor, and remove it from `meta`. Store `clip_emb` as `(n, 768)`.
- `make_item` (laya_train.py:293) returns `{q, state_ids, target, qtype, k}`. `items_from_rows` (laya_train.py:317) drops unknown row fields, so `clip_emb` must be attached in a copy of it.
- `train_model` (laya_train.py:649) calls module-level `encode_item` (line 696), `collate_items` (698), and `_forward` (699). A subclass cannot intercept these, so copy the loop (about 90 lines) into a vision variant. Do not monkeypatch the module.
- **Loss:** use `soft_ce_loss` (laya_train.py:493) first. The default `loss` is `"rlcd"` (line 99), whose Gaussian noise (sigma 0.4 to 0.1, lines 101-102) is comparable to the learned term. Set `loss="soft-ce"` explicitly.
- **noul:** always 2 markers, rendered as `"false: ..."` and `"true: ..."` (`render_options` noul branch). Embed those two strings like any option.
- `save_checkpoint` (laya_train.py:590) saves the full `state_dict`. `load_checkpoint` (laya_train.py:573) loads into a plain `DecisionModel` with `strict=True`, so `a` would fail. A vision loader is needed.

## 4. Inference bypasses and the thin wrapper

- `Agent._infer` (laya_agent.py:1207) passes five positional args plus position and option ids. It cannot pass `option_sim`, so write a separate wrapper.
- `accelerate()` (laya_agent.py:866) sets `self.model.forward = self._fast.forward` (lines 899-900), bypassing the subclass. `FastLaya` is not in `laya_src` (unverified).
- `set_backend` and `compile` (laya_agent.py:850-865, `_orig_mod` at 861) rewrap the model. Whether the subclass survives is unverified.
- `_pad_cuda_compile_batch` (laya_agent.py:267) pads only `input_ids` and `attention_mask`, so `option_sim` is unaffected.
- The wrapper must: load weights with `strict=True`; build sequences and `option_sim` with `option_order`; call the model under `no_grad` with `option_sim`; then divide by `T = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])` (laya_agent.py:1341), softmax, `unpermute_probs` (line 1350), and `answer_confidence` (laya_common.py:779).
- **Calibration:** `fit_temperature_map` (laya_calibrate.py:250) consumes `(qtype, logits, target, k)`. `calibration_records` (laya_train.py:739) and `records_from_labeled` (laya_calibrate.py:465-498, `collate_items` at line 487) bypass the subclass. Refit temperatures after training.

## 5. Design 2 (visual prefix via `inputs_embeds`)

- The encoder is called with `input_ids` only: laya_common.py:612 (sequential) and 616-618 (parallel). `inputs_embeds` appears nowhere in `laya_src`. Whether the HF encoder accepts it is **unverified**.
- Changes needed:
  - Insert K prefix slots after CLS. Index 0 must stay CLS, since `pooled = h[:, 0]` (line 651).
  - Shift every `marker_pos` by K (built in `build_head`, line 225).
  - Extend `attention_mask` with K ones.
  - Sequential layout: default positions shift with the input.
  - Parallel layout: `parallel_layout` (line 279) starts positions at `markers[0]`, so the prefix needs option id 0 and positions before the first marker.
  - Budgets: `state_room` (line 350) and the `max_len` checks in `make_item` and `_encode_state` need K.
- Head width is `d = encoder.config.hidden_size` (`DecisionModel.__init__`, line 577). A 768-d input needs a projection if `d != 768`. The published `d` is unverified.
- Risks: positions shift for every checkpoint, so a full re-finetune is needed. Parallel layout needs transformers 5 (`uses_parallel_layout`, line 311).

## 6. Test plan

- **T1 identity:** with `a = 0`, the copied forward matches `DecisionModel.forward` on logits and `act_logits` using a tiny stub encoder. Also check that a constant added to all options leaves probabilities unchanged.
- **T2 similarity permutation:** for order `o`, assert `option_sim[s] == sim_canon[o[s]]` and `target[s] == target[o[s]]`. Mutation check: permuting the target but not `option_sim` must fail.
- **T3 end-to-end invariance:** with `option_layout="parallel"` and a fixed clip, probabilities unpermuted by `o` must match the identity order. Skip for sequential.
- **T4 padding:** for `kmax > k`, padded slots get `-1e4` and zero loss. `collate` removes `option_sim` from `meta`.
- **T5 wiring:** the forward raises if `option_sim` is missing, which catches `_infer`, `collate_items`, and `records_from_labeled` bypasses. Save and load round-trip with `strict=True`.
