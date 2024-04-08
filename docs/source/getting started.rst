Getting Started
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

To install the cutting edge version from the main branch of this repo, run:

.. code-block:: console

   git clone https://github.com/anonymousLLM/AgentKit && cd AgentKit
   pip install -e .

Creating and running a basic graph
----------------

The basic building block in AgentKit is a node, containing a natural language prompt for a specific subtask. The nodes are linked together by the dependency specifications, which specify the order of evaluation. Different arrangements of nodes can represent different different logic and throught processes.

At inference time, AgentKit evaluates all nodes in specified order as a directed acyclic graph (DAG).

.. code-block:: python
   :linenos:

   from agentkit import Graph, BaseNode

   import agentkit.llm_api

   LLM_API_FUNCTION = agentkit.llm_api.get_query("gpt-4")

   graph = Graph()

   subtask1 = "What are the pros and cons for using LLM Agents for Game AI?" 
   node1 = BaseNode(subtask1, subtask1, graph, LLM_API_FUNCTION)
   graph.add_node(node1)

   subtask2 = "Give me an outline for an essay titled 'LLM Agents for Games'." 
   node2 = BaseNode(subtask2, subtask2, graph, LLM_API_FUNCTION)
   graph.add_node(node2)

   subtask3 = "Now, write a full essay on the topic 'LLM Agents for Games'."
   node3 = BaseNode(subtask3, subtask3, graph, LLM_API_FUNCTION)
   graph.add_node(node3)


   graph.add_edge(subtask1, subtask2)
   graph.add_edge(subtask1, subtask3)
   graph.add_edge(subtask2, subtask3)

   result = graph.evaluate() # outputs a dictionary of prompt, answer pairs

``LLM_API_FUNCTION`` can be any LLM querying function that takes ``msg:list`` and ``shrink_idx:int``, and outputs ``llm_result:str`` and ``usage:dict``. Where ``msg`` is a prompt (`OpenAI format`_ by default), and ``shrink_idx:int`` is an index at which the LLM should reduce the length of the prompt in case of overflow. 

AgentKit tracks token usage of each node through the ``LLM_API_FUNCTION`` with:

.. code-block:: python

   usage = {
      'prompt': $prompt token counts,
      'completion': $completion token counts,
   }



.. _OpenAI format: https://platform.openai.com/docs/guides/text-generation/chat-completions-api