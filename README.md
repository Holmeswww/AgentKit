<div align="center">
<img src="https://github.com/Holmeswww/AgentKit/raw/main/imgs/AgentKit.png" width="350px">

**AgentKit: Flow Engineering with Graphs, not Coding**

[[Arxiv Paper]](https://arxiv.org/abs/2404.11483)
[[PDF]](https://arxiv.org/pdf/2404.11483.pdf)
[[Docs]](https://agentkit.readthedocs.io/)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/agentkit-llm)](https://pypi.org/project/agentkit-llm/)
[![PyPI](https://img.shields.io/pypi/v/agentkit-llm)](https://pypi.org/project/agentkit-llm/)
[![PyPI Status](https://static.pepy.tech/badge/agentkit-llm)](https://pepy.tech/project/agentkit-llm)
[![Docs](https://readthedocs.org/projects/agentkit/badge/?version=latest)](https://agentkit.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/holmeswww/agentkit)](https://github.com/holmeswww/AgentKit/blob/main/LICENSE)
______________________________________________________________________
![](https://github.com/Holmeswww/AgentKit/raw/main/imgs/teaser.png)
</div>

<img src="https://github.com/Holmeswww/AgentKit/raw/main/imgs/AgentKit.png" width="65px"> offers a unified framework for explicitly constructing a complex human "thought process" from simple natural language prompts.
The user puts together chains of *nodes*, like stacking LEGO pieces. The chains of nodes can be designed to explicitly enforce a naturally *structured* "thought process".

Different arrangements of nodes could represent different functionalities, allowing the user to integrate various functionalities to build multifunctional agents.

A basic agent could be implemented as simple as a list of prompts for the subtasks and therefore could be designed and tuned by someone *without any programming experience*.


# Contents

- [Installation](#Installation)
- [Getting Started](#Getting-Started)
- [Using Built-in LLM_API](#Built-in-LLM-API)
- [Using AgentKit without Programming Experience](#Using-AgentKit-without-Programming-Experience)
- [Node Components](#Node-Components)
- [Commonly Asked Questions](#Commonly-Asked-Questions)
- [Citing AgnetKit](#Citing-AgentKit)

# Installation

Installing the AgentKit stable version is as simple as:

```bash
pip install agentkit-llm
```

To install AgentKit with wandb:

```bash
pip install agentkit-llm[logging]
```

To install AgentKit with built-in LLM-API support:

```bash
pip install agentkit-llm[all]
```

Otherwise, to install the cutting edge version from the main branch of this repo, run:

```bash
git clone https://github.com/holmeswww/AgentKit && cd AgentKit
pip install -e .
```

# Getting Started

The basic building block in AgentKit is a node, containing a natural language prompt for a specific subtask. The nodes are linked together by the dependency specifications, which specify the order of evaluation. Different arrangements of nodes can represent different different logic and throught processes.

At inference time, AgentKit evaluates all nodes in specified order as a directed acyclic graph (DAG).

```python
import agentkit
from agentkit import Graph, BaseNode

import agentkit.llm_api

LLM_API_FUNCTION = agentkit.llm_api.get_query("gpt-4-turbo")

LLM_API_FUNCTION.debug = True # Disable this to enable API-level error handling-retry

graph = Graph()

subtask1 = "What are the pros and cons for using LLM Agents for Game AI?" 
node1 = BaseNode(subtask1, subtask1, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt(), verbose=True)
graph.add_node(node1)

subtask2 = "Give me an outline for an essay titled 'LLM Agents for Games'." 
node2 = BaseNode(subtask2, subtask2, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt(), verbose=True)
graph.add_node(node2)

subtask3 = "Now, write a full essay on the topic 'LLM Agents for Games'."
node3 = BaseNode(subtask3, subtask3, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt(), verbose=True)
graph.add_node(node3)

# add dependencies between nodes
graph.add_edge(subtask1, subtask2)
graph.add_edge(subtask1, subtask3)
graph.add_edge(subtask2, subtask3)

result = graph.evaluate() # outputs a dictionary of prompt, answer pairs
```

``LLM_API_FUNCTION`` can be any LLM API function that takes ``msg:list`` and ``shrink_idx:int``, and outputs ``llm_result:str`` and ``usage:dict``. Where ``msg`` is a prompt ([OpenAI format](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) by default), and ``shrink_idx:int`` is an index at which the LLM should reduce the length of the prompt in case of overflow. 

AgentKit tracks token usage of each node through the ``LLM_API_FUNCTION`` with:
```python
usage = {
    'prompt': $prompt token counts,
    'completion': $completion token counts,
}
```

# Built-in LLM-API

The built-in `agentkit.llm_api` functions require installing with `[all]` setting. See [the installation guide](#Installation) for details.

Currently, the built-in API supports OpenAI and Anthropic, see https://pypi.org/project/openai/ and https://pypi.org/project/anthropic/ for details.

To use the OpenAI models, set environment variables `OPENAI_KEY` and `OPENAI_ORG`. Alternatively, you can put the openai 'key' and 'organization' in the first 2 lines of `~/.openai/openai.key`.

To use the Azure OpenAI models, set environment variables `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`, and `AZURE_DEPLOYMENT_NAME`. Alternatively, you can store the Azure OpenAI API key, API version, Azure endpoint, and deployment name in the first 4 lines of `~/.openai/azure_openai.key`.

To use the Anthropic models, set environment variable `ANTHROPIC_KEY`. Alternatively, you can put the anthropic 'key' in 3rd line of `~/.openai/openai.key`.

# Using AgentKit without Programming Experience

First, follow [the installation guide](#Installation) to install AgentKit with `[all]` setting.

Then, set environment variables `OPENAI_KEY` and `OPENAI_ORG` to be your OpenAI key and org_key.

Finally, run the following to evoke the command line interface (CLI):

```bash
git clone https://github.com/holmeswww/AgentKit && cd AgentKit
cd examples/prompt_without_coding
python generate_graph.py
```
![](https://github.com/Holmeswww/AgentKit/raw/main/imgs/screenshot.png)

# Node Components

![](https://github.com/Holmeswww/AgentKit/raw/main/imgs/node_archi.png)
Inside each node (as shown to the left of the figure), AgentKit runs a built-in flow that **preprocesses** the input (Compose), queries the LLM with a preprocessed input and prompt $q_v$, and optionally **postprocesses** the output of the LLM (After-query).

To support advanced capabilities such as branching, AgentKit offers API to dynamically modify the DAG at inference time (as shown to the right of the figure). Nodes/edges could be dynamically added or removed based on the LLM response at some ancestor nodes.

# Commonly Asked Questions

**Q:** I'm using the default `agentkit.llm_api`, and `graph.evaluate()` seems to be stuck.

**A:** The LLM_API function catches and retries all API errors by default. Set `verbose=True` for each node to see which node you are stuck on, and `LLM_API_FUNCTION.debug=True` to see what error is causing the error.

# Citing AgentKit
```bibtex
@article{wu2024agentkit,
    title={AgentKit: Flow Engineering with Graphs, not Coding}, 
    author={Yue Wu and Yewen Fan and So Yeon Min and Shrimai Prabhumoye and Stephen McAleer and Yonatan Bisk and Ruslan Salakhutdinov and Yuanzhi Li and Tom Mitchell},
    year={2024},
    journal={arXiv preprint arXiv:2404.11483}
}
```

# Star History

<a href="https://star-history.com/#holmeswww/agentkit&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=holmeswww/agentkit&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=holmeswww/agentkit&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=holmeswww/agentkit&type=Date" />
 </picture>
</a>
