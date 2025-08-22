from __future__ import annotations

from typing import Any

import torch  # type: ignore
from torch import Tensor  # type: ignore

from inspect_ai.model._providers.hf import (
    HuggingFaceAPI,
    batched_generate,
    GenerateInput,
    extract_logprobs,
    chat_completion_assistant_message,
)
from inspect_ai.model._providers.util import ChatAPIHandler, HFHandler
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    Logprobs,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo


class HFPrefillAPI(HuggingFaceAPI):
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Determine if last message is an assistant prefill
        prefill_text: str = ""
        use_prefill = False
        if len(input) > 0 and getattr(input[-1], "role", None) == "assistant":
            last_content = input[-1].content
            if isinstance(last_content, str):
                prefill_text = last_content
                use_prefill = True

        # Build the exact generation input string
        if use_prefill:
            ctx_plus_header = self.hf_chat(input[:-1], tools)
            gen_input = ctx_plus_header + prefill_text
        else:
            gen_input = self.hf_chat(input, tools)
            ctx_plus_header = gen_input

        # Prepare tokenizer/generator/decoder as in HF provider
        assert isinstance(self.tokenizer_call_args, dict)
        import functools

        tokenizer = functools.partial(
            self.tokenizer,  # type: ignore[misc]
            return_tensors="pt",
            padding=True,
            **self.tokenizer_call_args,
        )

        kwargs: dict[str, Any] = dict(do_sample=True)
        if config.max_tokens is not None:
            kwargs["max_new_tokens"] = config.max_tokens
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        if config.logprobs is not None:
            kwargs["output_logits"] = config.logprobs
        if self.hidden_states is not None:
            kwargs["output_hidden_states"] = self.hidden_states
        if config.stop_seqs is not None:
            from transformers.generation import StopStringCriteria  # type: ignore

            stopping_criteria = [StopStringCriteria(self.tokenizer, config.stop_seqs)]
            kwargs["stopping_criteria"] = stopping_criteria
        kwargs["return_dict_in_generate"] = True

        generator = functools.partial(self.model.generate, **kwargs)  # type: ignore[attr-defined]

        decoder = functools.partial(
            self.tokenizer.batch_decode,  # type: ignore[misc]
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Generate from after the prefill (gen_input already includes prefill tokens)
        response = await batched_generate(
            GenerateInput(
                input=gen_input,
                device=self.model.device,  # type: ignore[attr-defined]
                tokenizer=tokenizer,
                generator=generator,
                decoder=decoder,
                batch_size=config.max_connections or self.max_connections(),
            )
        )

        # Gather output token logprobs (as Inspect normally returns)
        final_logprobs = None
        if config.logprobs is not None:
            final_logprobs = extract_logprobs(
                response=response, top=config.top_logprobs, tokenizer=self.tokenizer  # type: ignore[misc]
            )

        handler: ChatAPIHandler | None = (
            HFHandler(self.model_name) if len(tools) > 0 else None
        )
        choice = ChatCompletionChoice(
            message=chat_completion_assistant_message(
                response, tools, handler, self.model_name
            ),
            logprobs=(Logprobs(content=final_logprobs) if final_logprobs is not None else None),
        )

        output = ModelOutput(
            model=self.model_name,
            choices=[choice],
            usage=ModelUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
            ),
            time=response.time,
            metadata={"hidden_states": response.hidden_states},
        )

        # Compute prompt/prefill logprobs via a single forward pass over gen_input
        try:
            prompt_ids, prompt_logprobs, prefill_ids, prefill_logprobs = self._score_prompt_and_prefill(
                ctx_plus_header, prefill_text if use_prefill else ""
            )
            # Store simple arrays in metadata for ease of downstream use
            if output.metadata is None:
                output.metadata = {}
            output.metadata.update(
                {
                    "prompt_token_ids": prompt_ids,
                    "prompt_logprobs": prompt_logprobs,
                    "prefill_token_ids": prefill_ids,
                    "prefill_logprobs": prefill_logprobs,
                }
            )
        except Exception:
            # Don't fail generation if scoring fails; leave metadata unset
            pass

        return output

    def _score_prompt_and_prefill(
        self, ctx_text: str, prefill_text: str
    ) -> tuple[list[int], list[float], list[int], list[float]]:
        full_text = ctx_text + prefill_text

        enc_full = self.tokenizer(  # type: ignore[operator]
            full_text, return_tensors="pt", padding=False
        )
        enc_ctx = self.tokenizer(  # type: ignore[operator]
            ctx_text, return_tensors="pt", padding=False
        )

        input_ids_full: Tensor = enc_full["input_ids"].to(self.model.device)  # type: ignore[attr-defined]
        attn_full: Tensor = enc_full["attention_mask"].to(self.model.device)  # type: ignore[attr-defined]
        ctx_len: int = enc_ctx["input_ids"].shape[1]

        with torch.inference_mode():
            logits: Tensor = self.model(  # type: ignore[call-arg]
                input_ids=input_ids_full, attention_mask=attn_full
            ).logits[0]
            logp: Tensor = torch.log_softmax(logits, dim=-1)

        # Prompt (ctx) token ids and logprobs (skip first token which has no predecessor)
        prompt_ids_tensor: Tensor = input_ids_full[0, :ctx_len]
        if ctx_len > 1:
            prompt_rows: Tensor = logp[: ctx_len - 1, :]
            prompt_logprobs_tensor: Tensor = prompt_rows.gather(
                -1, prompt_ids_tensor[1:].unsqueeze(-1)
            ).squeeze(-1)
            prompt_ids = prompt_ids_tensor.tolist()
            prompt_logprobs = prompt_logprobs_tensor.tolist()
        else:
            prompt_ids = prompt_ids_tensor.tolist()
            prompt_logprobs = []

        # Prefill token ids and logprobs (rows start at ctx_len-1)
        prefill_ids_tensor: Tensor = input_ids_full[0, ctx_len:]
        if prefill_ids_tensor.numel() > 0:
            prefill_rows: Tensor = logp[
                ctx_len - 1 : ctx_len - 1 + prefill_ids_tensor.shape[0], :
            ]
            prefill_logprobs_tensor: Tensor = prefill_rows.gather(
                -1, prefill_ids_tensor.unsqueeze(-1)
            ).squeeze(-1)
            prefill_ids = prefill_ids_tensor.tolist()
            prefill_logprobs = prefill_logprobs_tensor.tolist()
        else:
            prefill_ids = []
            prefill_logprobs = []

        return prompt_ids, prompt_logprobs, prefill_ids, prefill_logprobs


