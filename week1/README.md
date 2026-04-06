# Overview
This is my notes on the weekly reading materials and assignments.  
The code in this repo contains my completed solutions to the assignments.  
The original assignment can be found [here](https://github.com/mihail911/modern-software-dev-assignments/tree/master/week1).



# Note of Slides

## Mon 9/22: [Introduction and how an LLM is made](https://docs.google.com/presentation/d/1zT2Ofy88cajLTLkd7TcuSM4BCELvF9qQdHmlz33i4t0/edit?slide=id.p#slide=id.p)

- Coding with AI agents is not **Vibe Coding!**
- LLMs are only as good as you are.
- Read and review **a lot** of code: Learn to discern good from bad code, have a good taste.
- There are no established software patterns yet, figure out what works for you.

## Fri 9/26: [Power prompting for LLMs](https://docs.google.com/presentation/d/1MIhw8p6TLGdbQ9TcxhXSs5BaPf5d_h77QY70RHNfeGs/edit?slide=id.g37b974b8d4d_0_0#slide=id.g37b974b8d4d_0_0)

Just as moving from assembly to high-level languages was a paradigm shift in how we express logic to machines, prompting is the next evolution — a new layer of abstraction where intent matters more than syntax.

**Zero-Shot Prompting:**  
Ask the model a question directly without any examples, relying solely on its pretrained knowledge to respond.

**K-Shot Prompting:**  
Provide K input/output examples in the prompt so the model learns the expected format and reasoning before answering.

**Chain-of-Thought Prompting:**  
Guide the model to reason step-by-step rather than jumping straight to an answer, useful for multi-step logic tasks.
- **Multi-Shot CoT:** Include examples that demonstrate reasoning steps, so the model learns to imitate the thought process.
- **Zero-Shot CoT:** No examples — just add a trigger phrase like "Let's think step by step" to elicit reasoning.

**Self-Consistency Prompting:**  
Sample the same question multiple times across different reasoning paths, then select the most frequent answer via majority vote to improve reliability and accuracy.

**Retrieval Augmented Generation (RAG):**  
Before generating a response, retrieve relevant documents from an external knowledge base and inject them into the prompt, allowing the model to answer based on up-to-date or private data while reducing hallucinations.

**Reflexion:**  
After receiving execution results or feedback, the model reflects on its mistakes and generates an improvement strategy before retrying — forming an "act → evaluate → reflect → act" loop.

### Best Practice
- Structure prompts clearly (separate role, task, and format). For example, use XML-style tags to organize inputs:
    ```
  Here are the logs:
  <log>
  LOG MESSAGE
  </log>
  and the stack trace:
  <error>
  STACK TRACE
  </error>
    ```
- Be explicit about language, tech stack, libraries, and constraints.
- Assume zero background context — provide complete details as if talking to someone unfamiliar with the topic.


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