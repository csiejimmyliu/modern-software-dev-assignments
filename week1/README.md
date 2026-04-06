# Overview
This is my notes on the weekly reading materials and assignments.  
The code in this repo contains my completed solutions to the assignments.  
The original assignment can be found [here](https://github.com/mihail911/modern-software-dev-assignments/tree/master/week1).



# Note of Assignment

## K-shot Prompting
This was the hardest part of the Week 1 assignment for me. The success rate remained quite low even after experimenting with different strategies.

1. **Words with similar length to "sutatsptth"**: The model struggled with these — too long and confusing to reverse reliably.
2. **Random few-shot examples**: Still struggled to internalize the core concept of reversing. 
3. **Classic simple words** (apple, abc, helloworld): Produced the best results. These words likely appear so frequently in training data that the model has an intuitive sense of their letter order — making reversal more reliable.

I ended up going with strategy 3, blending in a few more complex words to push the model's generalization.

## Chain-of-Thought
Beyond instructing the model to think step by step, I also explicitly guided each stage of the reasoning process in the prompt.
Output format requires extra attention — the model would occasionally arrive at the correct answer but fail to present it in the expected format.

## Tool Calling
Clearly specifying the tool call format and explaining each parameter in the prompt made a significant difference.
That said, having only one callable tool felt a bit too easy — the model simply returns the only available option regardless of intent.

A more robust test would define multiple tools and vary the user prompt to confirm the model genuinely selects based on the request.
```
tools: output_every_func_return_type, count_func
user prompt 1: give me the function return types
user prompt 2: count the number of functions
```
This way we can verify that tool selection is actually driven by the user's intent, not just a process of elimination.

## Self-consistency Prompting
The self-consistency mechanism in this assignment lives in the code itself, not in anything we control through the prompt.
It works by sampling the same prompt multiple times and taking the most frequent answer as the final output — essentially a majority vote over multiple runs.

## RAG (Retrieval-Augmented Generation)
**Data Flow**

1. **Load** — Read `api_docs.txt` into CORPUS as the knowledge base.
2. **Retrieve** — Context Provider selects relevant documents from CORPUS.
3. **Augment** — Combine the selected docs and the question into a single prompt.
4. **Generate** — Send the prompt to the LLM and obtain the generated code.
5. **Validate** — Check whether the output contains all required code snippets.

> **Note:** In this assignment, Retrieve simply hardcodes `corpus[0]`.
> In production, this step uses embedding-based vector search to surface the most relevant documents automatically.

## Reflexion
The LLM iteratively improves its own code based on failure feedback — rather than regenerating from scratch, it reflects on what went wrong and produces a targeted fix.

**Data Flow**

1. **Generate** — LLM writes the initial `is_valid_password()` implementation.
2. **Evaluate** — Run test cases and collect failure diagnostics.
3. **Reflect** — Feed the buggy code along with the failure messages back to the LLM.
4. **Regenerate** — LLM produces a corrected implementation.
5. **Validate** — Check whether the improved code passes all test cases.

> **Note:** In this assignment, Evaluate relies on hardcoded test cases.
> In production, this step could be delegated to an LLM judge for more flexible evaluation.