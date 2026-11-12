"""GRPOTrainer with two alignment-preserving terms that TRL does not provide.

1. Anchor SFT (``anchor_coef``, ``anchor_every``)
   Adds ``anchor_coef * NLL(anchor batch)`` to the RL loss. ``anchor_every=1`` is the continuous
   "PPO-ptx"-style mix (Ouyang et al. 2022); ``anchor_every=N`` with a larger coefficient gives
   interleaved SFT checkpoints, where every N-th optimizer step also pulls the policy toward anchor data.

2. On-policy self-distillation from a conduct-spec teacher (``sd_coef``)
   The teacher is the *frozen initial model* prompted with a conduct spec as its system prompt; the
   student sees no spec. For every sampled completion token we add the per-token reverse-KL reward
   ``sd_coef * (log p_teacher(y_t) - log p_student(y_t))`` to the GRPO advantage, i.e. on-policy
   distillation folded into the policy-gradient update. This is the approach Thinking Machines
   describes for Inkling 2 Small ("learn from a teacher prompted with a conduct spec").
   With an empty spec it reduces to a per-token KL-to-reference penalty in the reward, which makes a
   clean ablation: *what does the spec add beyond plain KL?*
"""

from __future__ import annotations

import itertools
import random

import torch
from trl import GRPOTrainer
from trl.trainer import grpo_trainer as _g


class AlignedGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        anchor_data: list[dict] | None = None,
        anchor_coef: float = 0.0,
        anchor_every: int = 1,
        anchor_batch_size: int = 4,
        anchor_max_length: int = 768,
        sd_coef: float = 0.0,
        sd_clip: float = 5.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_coef, self.anchor_every = anchor_coef, max(1, anchor_every)
        self.sd_coef, self.sd_clip = sd_coef, sd_clip
        self._teacher = None
        if anchor_coef > 0:
            if not anchor_data:
                raise ValueError("anchor_coef > 0 requires anchor data (recipe.anchor.path)")
            self._anchor_iter = self._make_anchor_iter(anchor_data, anchor_batch_size, anchor_max_length)
        if sd_coef > 0 and self.ref_model is None and not _g.is_peft_model(self.model):
            # Full fine-tuning with beta == 0: TRL built no reference model, so load a frozen copy.
            self._teacher = _g.create_model_from_path(
                _g.get_config_model_id(self.model.config), dtype=self.model.dtype
            ).to(self.accelerator.device).eval()
            self._teacher.requires_grad_(False)

    # ---------------------------------------------------------------- anchor SFT
    def _make_anchor_iter(self, data, bs, max_len):
        tok = self.processing_class
        tmpl = self.chat_template_kwargs if hasattr(self, "chat_template_kwargs") else {}
        tmpl = tmpl or {}
        examples = []
        for ex in data:
            msgs = [{"role": "user", "content": ex["prompt"]}]
            p_txt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **tmpl)
            full_txt = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": ex["response"]}], tokenize=False, **tmpl
            )
            p_ids = tok(p_txt, add_special_tokens=False)["input_ids"]
            full = tok(full_txt, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * min(len(p_ids), len(full)) + full[len(p_ids):]
            if any(l != -100 for l in labels):
                examples.append((full, labels))
        if len(examples) < bs:
            raise ValueError(f"anchor data yielded {len(examples)} usable examples; need >= batch_size={bs}")
        rng = random.Random(0)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def gen():
            for epoch in itertools.count():
                rng.shuffle(examples)
                for i in range(0, len(examples) - bs + 1, bs):
                    batch = examples[i : i + bs]
                    L = max(len(x) for x, _ in batch)
                    ids = torch.full((bs, L), pad)
                    lab = torch.full((bs, L), -100)
                    att = torch.zeros((bs, L), dtype=torch.long)
                    for j, (x, y) in enumerate(batch):
                        ids[j, : len(x)] = torch.tensor(x)
                        lab[j, : len(y)] = torch.tensor(y)
                        att[j, : len(x)] = 1
                    yield ids, att, lab

        return gen()

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if self.anchor_coef > 0 and model.training and self.state.global_step % self.anchor_every == 0:
            ids, att, lab = (t.to(self.accelerator.device) for t in next(self._anchor_iter))
            nll = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss = loss + self.anchor_coef * nll / self.current_gradient_accumulation_steps
            self._metrics["train"]["anchor/nll"].append(nll.detach().item())
        return loss

    # ---------------------------------------------------------------- self-distillation
    @torch.no_grad()
    def _teacher_logps(self, input_ids, attention_mask, logits_to_keep):
        if self._teacher is not None:
            m = self._teacher
            return self._get_per_token_logps_and_entropies(m, input_ids, attention_mask, logits_to_keep)[0]
        if self.ref_model is not None:
            return self._get_per_token_logps_and_entropies(self.ref_model, input_ids, attention_mask, logits_to_keep)[0]
        unwrapped = self.accelerator.unwrap_model(self.model)
        with _g.use_adapter(unwrapped, adapter_name="ref" if "ref" in unwrapped.peft_config else None):
            return self._get_per_token_logps_and_entropies(self.model, input_ids, attention_mask, logits_to_keep)[0]

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if self.sd_coef <= 0 or not self.model.training:
            return out
        device = self.accelerator.device
        c_ids, c_mask = out["completion_ids"], out["completion_mask"]
        T = c_ids.size(1)
        tok = self.processing_class
        enc = tok(
            text=[x["teacher_prompt"] for x in inputs], return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False,
        ).to(device)
        t_logps = self._teacher_logps(
            torch.cat([enc["input_ids"], c_ids], 1), torch.cat([enc["attention_mask"], c_mask], 1), T
        )
        s_logps = out.get("old_per_token_logps")
        if s_logps is None:
            with torch.no_grad():
                s_logps = self._get_per_token_logps_and_entropies(
                    self.model,
                    torch.cat([out["prompt_ids"], c_ids], 1),
                    torch.cat([out["prompt_mask"], c_mask], 1),
                    T,
                )[0]
        tok_adv = (self.sd_coef * (t_logps - s_logps)).clamp(-self.sd_clip, self.sd_clip) * c_mask
        adv = out["advantages"]
        out["advantages"] = (adv.unsqueeze(1) if adv.dim() == 1 else adv) + tok_adv
        rkl = ((s_logps - t_logps) * c_mask).sum() / c_mask.sum().clamp(min=1)
        self._metrics["train"]["self_distill/reverse_kl"].append(self.accelerator.gather(rkl).mean().item())
        return out
