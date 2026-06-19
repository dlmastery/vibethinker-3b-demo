"""
VibeThinker inference engine.

Wraps the VibeThinker-3B (or 1.5B) checkpoint and exposes two inference modes:

  * Non-CLR  -> a single greedy/sampled generation pass (the model's default
                behaviour described in the official README).
  * CLR      -> a faithful community reconstruction of "Claim-Level Reliability
                Assessment" (Xu et al., 2026). The official repo describes CLR
                as a test-time scaling strategy for *answer-verifiable* reasoning
                but does not ship the algorithm, so we reconstruct it as:
                    1. sample N independent solutions,
                    2. extract the final \\boxed{} answer of each,
                    3. take the self-consistency majority ("cons@n" in the repo),
                    4. run a focused claim-verification pass on the leading
                       answer (re-deriving the critical anchors rather than the
                       whole verbose trace),
                    5. pick the final answer from votes + the verification verdict
                       and report a reliability score.
                This mirrors the repo's own `cons@n` selection plus the paper's
                "isolate critical logical anchors" verification idea.

All recommended generation parameters come from the official README:
    temperature = 0.6 or 1.0, top_p = 0.95, top_k = -1 (disabled),
    max_new_tokens up to 40960, dtype = bfloat16.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread
from typing import Iterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


class _EventStop(StoppingCriteria):
    """Stop generation as soon as `event` is set (e.g. client disconnected)."""

    def __init__(self, event: "threading.Event"):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


def _criteria(event):
    return StoppingCriteriaList([_EventStop(event)]) if event is not None else None

MODEL_DIR = str(Path(__file__).parent / "models" / "VibeThinker-3B")
MODEL_NAME = "VibeThinker-3B"

# A single GPU can only run one generation at a time; serialise everything.
_GPU_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Answer extraction helpers
# --------------------------------------------------------------------------- #
def _extract_boxed(text: str) -> Optional[str]:
    """Return the content of the LAST \\boxed{...}, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if out else None


def extract_answer(text: str) -> Optional[str]:
    """Best-effort final-answer extraction for answer-verifiable tasks."""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return _normalize(boxed)
    # Fallback: "final answer is X" patterns
    m = re.findall(r"final answer[^\n\d\-]*[:\s]\s*([\-\d./,]+)", text, flags=re.I)
    if m:
        return _normalize(m[-1])
    # Last resort: last standalone number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return _normalize(nums[-1]) if nums else None


def _normalize(ans: str) -> str:
    ans = ans.strip().strip("$").strip()
    ans = ans.rstrip(".")
    # collapse trivial latex wrappers
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = re.sub(r"\s+", "", ans)
    return ans


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
@dataclass
class GenParams:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192
    do_sample: bool = True


@dataclass
class ClrCandidate:
    index: int
    answer: Optional[str]
    text: str
    tokens: int


@dataclass
class ClrResult:
    final_answer: Optional[str]
    candidates: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)
    reliability: float = 0.0
    verification_verdict: Optional[str] = None
    verification_answer: Optional[str] = None


class VibeThinkerEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self._load_lock = threading.Lock()
        self.load_error: Optional[str] = None

    # -- lifecycle -------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def device_info(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    def load(self):
        if self.ready:
            return
        with self._load_lock:
            if self.ready:
                return
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    low_cpu_mem_usage=True,
                    torch_dtype="bfloat16",
                    device_map="auto",
                )
                self.model.eval()
                # config.json ships use_cache=False (training leftover); KV cache
                # is essential for usable generation speed.
                self.model.config.use_cache = True
                # ChatML turns end with <|im_end|>; the base eos is <|endoftext|>.
                # Stop on either so generation halts at end of the assistant turn.
                im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                eos = {self.tokenizer.eos_token_id}
                if isinstance(im_end, int) and im_end >= 0:
                    eos.add(im_end)
                self.eos_ids = sorted(i for i in eos if i is not None)
                self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            except Exception as e:  # surface load errors to the UI
                self.load_error = repr(e)
                raise

    # -- prompt building -------------------------------------------------- #
    def _build_inputs(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    # -- streaming single pass (non-CLR) --------------------------------- #
    def stream(self, prompt: str, params: GenParams, stop_event=None) -> Iterator[str]:
        """Yield decoded text chunks as the model generates (non-CLR mode)."""
        self.load()
        with _GPU_LOCK:
            inputs = self._build_inputs(prompt)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=0,  # disabled, per README (top_k=-1 in vLLM)
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
                use_cache=True,
                stopping_criteria=_criteria(stop_event),
            )
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            try:
                for chunk in streamer:
                    if chunk:
                        yield chunk
            finally:
                thread.join()

    # -- single full generation (used internally by CLR) ----------------- #
    def _generate_once(self, prompt: str, params: GenParams) -> str:
        inputs = self._build_inputs(prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=params.do_sample,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=0,
                eos_token_id=self.eos_ids,
                pad_token_id=self.pad_id,
                use_cache=True,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    # -- internal: stream one generation, yielding chunks ---------------- #
    def _stream_one(self, prompt: str, params: GenParams, stop_event=None):
        """Generator yielding text chunks for a single pass (no lock; caller holds it)."""
        inputs = self._build_inputs(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=params.max_new_tokens,
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=0,
            eos_token_id=self.eos_ids,
            pad_token_id=self.pad_id,
            use_cache=True,
            stopping_criteria=_criteria(stop_event),
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()

    # -- CLR: Claim-Level Reliability Assessment (reconstruction) -------- #
    def clr_stream(self, prompt: str, params: GenParams, n_samples: int = 4,
                   stop_event=None):
        """
        Yield CLR progress events as dicts. See module docstring for the method.
        """
        self.load()
        with _GPU_LOCK:
            candidates: list[ClrCandidate] = []
            yield {"type": "phase", "phase": "sampling", "total": n_samples}

            # 1. Sample N independent solutions (slightly higher temperature
            #    to encourage diversity, matching the repo's cons@n setup).
            sample_params = GenParams(
                temperature=max(params.temperature, 1.0),
                top_p=params.top_p,
                max_new_tokens=params.max_new_tokens,
            )
            for i in range(n_samples):
                if stop_event is not None and stop_event.is_set():
                    return
                yield {"type": "sample_start", "index": i}
                text_parts = []
                for chunk in self._stream_one(prompt, sample_params, stop_event):
                    text_parts.append(chunk)
                    yield {"type": "sample_token", "index": i, "text": chunk}
                full = "".join(text_parts)
                ans = extract_answer(full)
                cand = ClrCandidate(
                    index=i, answer=ans, text=full, tokens=self.count_tokens(full)
                )
                candidates.append(cand)
                yield {"type": "sample_done", "index": i, "answer": ans,
                       "tokens": cand.tokens}

            # 2. Self-consistency tally over extracted answers (the repo's cons@n).
            valid = [c.answer for c in candidates if c.answer is not None]
            votes = dict(Counter(valid))
            leader = None
            if votes:
                leader = max(votes.items(), key=lambda kv: kv[1])[0]
            yield {"type": "votes", "votes": votes, "leader": leader}

            # 3. Claim-level verification: isolate the critical anchors of the
            #    leading answer and re-check them, rather than re-reading the
            #    full verbose trace.
            verdict = None
            verify_answer = None
            if leader is not None:
                anchor_text = self._anchor_excerpt(
                    next(c for c in candidates if c.answer == leader).text
                )
                vprompt = self._verification_prompt(prompt, leader, anchor_text)
                yield {"type": "phase", "phase": "verifying", "candidate": leader}
                vparts = []
                vparams = GenParams(
                    temperature=0.3, top_p=0.95,
                    max_new_tokens=min(params.max_new_tokens, 6144),
                )
                for chunk in self._stream_one(vprompt, vparams, stop_event):
                    vparts.append(chunk)
                    yield {"type": "verify_token", "text": chunk}
                vfull = "".join(vparts)
                verify_answer = extract_answer(vfull)
                up = vfull.upper()
                if "VERDICT: CORRECT" in up or "VERDICT:CORRECT" in up:
                    verdict = "CORRECT"
                elif "VERDICT: INCORRECT" in up or "VERDICT:INCORRECT" in up:
                    verdict = "INCORRECT"
                else:
                    verdict = "UNCERTAIN"
                yield {"type": "verify_done", "verdict": verdict,
                       "answer": verify_answer}

            # 4. Final selection + reliability score.
            final = leader
            n_votes = votes.get(leader, 0) if leader else 0
            reliability = (n_votes / n_samples) if n_samples else 0.0
            if verdict == "CORRECT":
                reliability = min(1.0, reliability + (1 - reliability) * 0.5)
            elif verdict == "INCORRECT":
                # verifier disagrees: trust its re-derived answer, lower reliability
                if verify_answer is not None and verify_answer != leader:
                    final = verify_answer
                reliability *= 0.5

            yield {
                "type": "final",
                "answer": final,
                "reliability": round(reliability, 3),
                "verdict": verdict,
                "votes": votes,
                "candidates": [
                    {"index": c.index, "answer": c.answer, "tokens": c.tokens}
                    for c in candidates
                ],
            }

    @staticmethod
    def _anchor_excerpt(text: str, head: int = 600, tail: int = 1200) -> str:
        """Condense a trace to its critical anchors: opening setup + final steps."""
        text = text.strip()
        if len(text) <= head + tail:
            return text
        return text[:head] + "\n...[reasoning condensed]...\n" + text[-tail:]

    @staticmethod
    def _verification_prompt(problem: str, answer: str, anchors: str) -> str:
        return (
            "You are a meticulous mathematics verifier. Independently check the "
            "proposed final answer to the problem below.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"PROPOSED FINAL ANSWER: {answer}\n\n"
            "Key claims / anchors from the candidate solution:\n"
            f"{anchors}\n\n"
            "Re-derive the critical steps yourself and decide whether the proposed "
            "final answer is correct. Be concise. Conclude with exactly one line "
            "'VERDICT: CORRECT' or 'VERDICT: INCORRECT', and put the correct final "
            "answer in \\boxed{}."
        )


# Module-level singleton used by the web server.
ENGINE = VibeThinkerEngine()
