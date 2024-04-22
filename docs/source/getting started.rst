Getting Started
=====

.. _installation:

Installation
------------

Installing the AgentKit stable version is as simple as:

.. code-block:: console

   pip install agentkit-llm

To install AgentKit with wandb:

.. code-block:: console

   pip install agentkit-llm[logging]

To install AgentKit with built-in LLM-API support:

.. code-block:: console

   pip install agentkit-llm[all]

Otherwise, to install the cutting edge version from the main branch of this repo, run:

.. code-block:: console

   git clone https://github.com/holmeswww/AgentKit && cd AgentKit
   pip install -e .

Creating and running a basic graph
----------------

The basic building block in AgentKit is a node, containing a natural language prompt for a specific subtask. The nodes are linked together by the dependency specifications, which specify the order of evaluation. Different arrangements of nodes can represent different different logic and throught processes.

.. figure:: https://github.com/Holmeswww/AgentKit/raw/main/imgs/teaser.png
    :scale: 80 %
    :alt: Illustration of what's possible with AgentKit

At inference time, AgentKit evaluates all nodes in specified order as a directed acyclic graph (DAG).

.. code-block:: python
   :linenos:

   import agentkit

   from agentkit import Graph, BaseNode

   import agentkit.llm_api

   LLM_API_FUNCTION = agentkit.llm_api.get_query("gpt-4-turbo")

   graph = Graph()

   subtask1 = "What are the pros and cons for using LLM Agents for Game AI?" 
   node1 = BaseNode(subtask1, subtask1, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt())
   graph.add_node(node1)

   subtask2 = "Give me an outline for an essay titled 'LLM Agents for Games'." 
   node2 = BaseNode(subtask2, subtask2, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt())
   graph.add_node(node2)

   subtask3 = "Now, write a full essay on the topic 'LLM Agents for Games'."
   node3 = BaseNode(subtask3, subtask3, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt())
   graph.add_node(node3)

   # add dependencies between nodes
   graph.add_edge(subtask1, subtask2)
   graph.add_edge(subtask1, subtask3)
   graph.add_edge(subtask2, subtask3)

   result = graph.evaluate() # outputs a dictionary of prompt, answer pairs

The built-in ``agentkit.llm_api`` functions require installing with ``[all]`` setting.

Currently, the built-in API supports OpenAI and Anthropic, see https://pypi.org/project/openai/ and https://pypi.org/project/anthropic/ for details.

To use the OpenAI models, set environment variables ``OPENAI_KEY`` and ``OPENAI_ORG``. Alternatively, you can put the openai 'key' and 'organization' in the first 2 lines of ``~/.openai/openai.key``.

To use the Azure OpenAI models, set environment variables ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_API_VERSION``, ``AZURE_OPENAI_ENDPOINT``, and ``AZURE_DEPLOYMENT_NAME``. Alternatively, you can store the Azure OpenAI API key, API version, Azure endpoint, and deployment name in the first 4 lines of ``~/.openai/azure_openai.key``.

To use the Anthropic models, set environment variable ``ANTHROPIC_KEY``. Alternatively, you can put the anthropic 'key' in 3rd line of ``~/.openai/openai.key``.

``LLM_API_FUNCTION`` can be any LLM querying function that takes ``msg:list`` and ``shrink_idx:int``, and outputs ``llm_result:str`` and ``usage:dict``. Where ``msg`` is a prompt (`OpenAI format`_ by default), and ``shrink_idx:int`` is an index at which the LLM should reduce the length of the prompt in case of overflow. 

AgentKit tracks token usage of each node through the ``LLM_API_FUNCTION`` with:

.. code-block:: python

   usage = {
      'prompt': prompt_token_count,
      'completion': completion_token_count,
   }



.. _OpenAI format: https://platform.openai.com/docs/guides/text-generation/chat-completions-api
