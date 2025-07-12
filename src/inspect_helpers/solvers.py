from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.model import CachePolicy, GenerateConfigArgs
from typing import Literal, Unpack
from src.inspect_helpers.datasets import PREFILL_KEY

@solver
def prefill_generate(
    prefill_message: str = None,
    tool_calls: Literal["loop", "single", "none"] = "loop",
    cache: bool | CachePolicy = False,
    **kwargs: Unpack[GenerateConfigArgs],
) -> Solver: 
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if prefill_message or state.metadata[PREFILL_KEY] is not None:
            if prefill_message:
                state.messages.append(ChatMessageAssistant(content=state.metadata[PREFILL_KEY]))  
            else:
                state.messages.append(ChatMessageAssistant(content=state.metadata[PREFILL_KEY]))
        return await generate(state, tool_calls=tool_calls, cache=cache, **kwargs)

    return solve