Dynamic Components
=====

.. figure:: https://github.com/Holmeswww/AgentKit/raw/main/imgs/node_archi.png
    :scale: 80 %
    :alt: Illustration of what's happening inside a node in AgentKit

    Each node in AgentKit takes outputs from its dependencies and outputs a string to complete a predefined subtask. The orange components (After-query) are optional and can be further customized with minimal programming through the AgentKit API. 
    **Right**: Nodes can be dynamically added/removed during the inference time. For example, the after-query operation of :math:`n_7` adds a conditional node :math:`n_{+}/n_{-}` based on a yes/no answer from the LLM to the node query. This induces conditional branching.

To support advanced capabilities such as branching, AgentKit offers API for user to dynamically modify the DAG at inference time (Figure right).
All dynamic modifications are *temporary* and will be reverted at the end of a graph traversal pass.
Note that modifications to nodes already evaluated in this pass are forbidden and will be automatically rejected.

Dynamic modifications are typically performed in the after-query phase of a node, where the user can add new nodes, remove existing nodes, or modify the connections between nodes.


Conditional Adding Nodes/Edges
----------------
Nodes/edges can be added conditionally based on the output of the LLM.
AgentKit provides helper functions to add nodes and edges during evaluation:

.. autofunction:: agentkit.graph.Graph.add_temporary_node
.. autofunction:: agentkit.graph.Graph.add_edge_temporary

The following example demonstrates how to add a conditional node based on a 'yes/no' answer from the LLM.
The after-query operation adds a conditional node based on the parsed JSON answer from the LLM.

.. code-block:: python

    class AddNodeAfterQuery(agentkit.after_query.JsonAfterQuery):

            def __init__(self):
            super().__init__()
            self.type = dict
            self.required_keys = ['answer']
            self.length = 1

        def post_process(self):

            json_dict = self.parse_json()[-1]

            if 'yes' in json_dict['answer'].lower():
                # If the answer is yes, add a 'yes' node
                add_node = SimpleDBNode(self.node.db['prompts']['yes_prompt'], questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db)
                self.node.graph.add_temporary_node(add_node)

                # Connect the 'yes' node to all the 'yes' dependencies
                for node in self.node.db['dependencies']['yes_dependencies']:
                    self.node.graph.add_edge_temporary(node.id, add_node.id)
                
                # Connect the 'yes' node to all nodes that 'depends_on_yes'
                for node in self.node.db['dependencies']['depends_on_yes']:
                    self.node.graph.add_edge_temporary(add_node.id, node.id)

            else:
                # If the answer is no, add a 'no' node
                add_node = SimpleDBNode(self.node.db['prompts']['no_prompt'], questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db)
                self.node.graph.add_temporary_node(add_node)

                # Connect the 'no' node to all the 'no' dependencies
                for node in self.node.db['dependencies']['no_dependencies']:
                    self.node.graph.add_edge_temporary(node.id, add_node.id)

                # Connect the 'no' node to all nodes that 'depends_on_no'
                for node in self.node.db['dependencies']['depends_on_no']:
                    self.node.graph.add_edge_temporary(add_node.id, node.id)

Conditional Skipping
----------------
In an agent setting, it is common to save computation by re-using outputs from the previous turn.
For example, if the plan does not need to be updated, we can skip all the nodes that are related to the planner.
AgentKit provides a helper function to skip certain nodes and re-use the saved outputs:

.. autofunction:: agentkit.graph.Graph.skip_nodes_temporary

The following example demonstrates how to skip nodes based on a 'yes/no' answer from the LLM.

.. code-block:: python

    class ReflectionAfterQuery(aq.JsonAfterQuery):
        
        def __init__(self):
            super().__init__()
            self.type = dict
            self.required_keys = ['update_plan']
            self.length = 1

        def post_process(self):

            json_dict = self.parse_json()[-1]

            if 'yes' not in json_dict['update_plan'].lower():
                # Skip all the nodes that are in the 'planner_nodes_to_skip' list if the planner does not need to be updated
                self.node.graph.skip_nodes_temporary(self.node.db["prompts"]["planner_nodes_to_skip"])


Conditional Branching
----------------
AgentKit allows users to manage branching in the graph by conditionally removing edges between nodes:

.. autofunction:: agentkit.graph.Graph.remove_edge_temporary

The following example demonstrates how to direct and manage the 'flow' of the graph between two pre-built leaf nodes, based on the output of the LLM.

.. code-block:: python

    class YesNoAfterQuery(agentkit.after_query.JsonAfterQuery):

            def __init__(self):
            super().__init__()
            self.type = dict
            self.required_keys = ['answer']
            self.length = 1

        def post_process(self):

            json_dict = self.parse_json()[-1]

            if 'yes' in json_dict['answer'].lower():
                # If the answer is yes, remove the edge between the current node and the 'no' node
                self.node.graph.remove_edge_temporary(self.node.id, self.node.db['dependencies']['no_node'])

            else:
                # If the answer is no, remove the edge between the current node and the 'yes' node
                self.node.graph.remove_edge_temporary(self.node.id, self.node.db['dependencies']['yes_node'])
