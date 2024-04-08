from collections import deque
import datetime
try:
    from wandb.sdk.data_types.trace_tree import Trace
except:
    print("wandb Trace seems to be missing. This is fine if you are not using wandb")
    pass

class Graph:
    """A class to represent a DAG.

    This class represents a DAG with nodes and edges. It is used to represent
    the dependencies between different nodes in a graph. The graph can be
    evaluated in a topological order. The graph can also have temporary nodes
    and edges. Temporary nodes and edges are used to represent dynamic changes
    in the graph during evaluation. Temporary nodes and edges are cleared after
    each evaluation. The evaluation processs can also be logged to wandb.

    Attributes:
        nodes (dict): A dictionary of all nodes in the graph.
        temporary_nodes (dict): A dictionary of all temporary nodes in the graph.
        temporary_edges (list): A list of all temporary edges in the graph.
        temporary_removed_edges (list): A list of all edges that have been temporarily removed.
        history (dict): A dictionary of all results from the graph.
        num_iter (int): The number of iterations the graph has gone through.
        history_list (list): A list of all results from the graph.
        order (list): A list of the order in which the nodes were evaluated.
        queue (list): A list of nodes that need to be evaluated.
        wandb_root_span (wandb.sdk.data_types.trace_tree.Trace): The root span for wandb logging.
        chain_span (wandb.sdk.data_types.trace_tree.Trace): The chain span for logging the current step.
    """
    def __init__(self):
        self.nodes = {}  # Dictionary to store all nodes
        self.temporary_nodes = {}
        self.temporary_edges = []
        self.temporary_removed_edges = []
        self.history = {}  # Dictionary to store all results
        self.num_iter = 0
        self.history_list = []
        self.order = []
        self.queue = []
        self.wandb_root_span = None
        self.chain_span = None
    
    def get_node_with_temporary(self, key):
        """Get a node from the graph.

        Args:
            key (str): The key of the node to get.
        
        Returns:
            Node: The node with the key.

        Raises:
            ValueError: If the key is not found in the graph.
        """

        if key in self.nodes.keys():
            return self.nodes[key]
        elif key in self.temporary_nodes.keys():
            return self.temporary_nodes[key]
        else:
            raise ValueError("Node ({}) not found in graph".format(key))

    def add_node(self, node):
        """Add a node to the graph.

        Args:
            node (Node): The node to add to the graph.
        
        Raises:
            AssertionError: If the node already exists in the graph.
        """
        assert node.key not in self.nodes.keys(), "Node ({}) already exists".format(node.key)
        self.nodes[node.key] = node

    def add_temporary_node(self, node):
        """Add a temporary node to the graph.

        Args:
            node (Node): The temporary node to add to the graph.
        
        Raises:
            AssertionError: If the node already exists in the graph.
        """
        assert node.key not in self.nodes.keys(), "Node ({}) already exists in permanent graph".format(node.key)
        assert node.key not in self.temporary_nodes.keys(), "Node ({}) already exists in temporary graph".format(node.key)
        self.temporary_nodes[node.key] = node
        
    def has_edge_with_temporary(self, from_key, to_key):
        """Check if there is an edge between two nodes in the graph.

        Args:
            from_key (str): The key of the node to check from.
            to_key (str): The key of the node to check to.
        
        Returns:
            bool: True if there is an edge between the two nodes, False otherwise.
        
        Raises:
            AssertionError: If the from_key is not found in the graph.
        """
        node_from = self.get_node_with_temporary(from_key)
        node_to = self.get_node_with_temporary(to_key)
        assert node_from is not None and node_to is not None, "Node ({}) not found in graph".format(from_key)
        return node_from in node_to.adjacent_from

    def add_edge(self, from_key, to_key, prepend=False):
        """Add an edge between two nodes in the graph.

        Args:
            from_key (str): The key of the node to add the edge from.
            to_key (str): The key of the node to add the edge to.
            prepend (bool): Whether to prepend the edge to the front of the adjacent_from list.
        
        Raises:
            ValueError: If the from_key or to_key is not found in the graph.
        """
        if from_key in self.nodes.keys() and to_key in self.nodes.keys():
            assert not self.has_edge_with_temporary(from_key, to_key), "Edge ({}) already exists".format((from_key, to_key))
            self.nodes[from_key].adjacent_to.append(self.nodes[to_key])
            if prepend:
                self.nodes[to_key].adjacent_from.insert(0, self.nodes[from_key])
            else:
                self.nodes[to_key].adjacent_from.append(self.nodes[from_key])
        else:
            raise ValueError("Node ({}, {}) not found in graph".format(from_key, to_key))
    
    def add_order(self, from_key, to_key):
        """Add an order (non-dependency edge) between two nodes in the graph.

        This function can be used to specify the order without adding an edge, 
        in order to avoid a node being evaluated before another node.


        Args:
            from_key (str): The key of the node to add the order from.
            to_key (str): The key of the node to add the order to.
        
        Raises:
            AssertionError: If the from_key or to_key is not found in the graph.
            AssertionError: If the edge already exists between the two nodes.
        
        Note:
            This function does not add an edge between the two nodes.
        """
        assert from_key in self.nodes.keys() and to_key in self.nodes.keys(), "Node not found in graph"
        assert not self.has_edge_with_temporary(from_key, to_key), "Edge {} already exists. No need to specify order".format((from_key, to_key))
        self.nodes[to_key].evaluate_after.append(self.nodes[from_key])

    def add_edge_temporary(self, from_key, to_key, prepend=False):
        """Add a temporary edge between two nodes in the graph.

        This function adds a temporary edge between the two nodes. Temporary edges
        are used to represent dynamic changes in the graph during evaluation. Temporary
        edges are cleared after each evaluation.

        Args:
            from_key (str): The key of the node to add the edge from.
            to_key (str): The key of the node to add the edge to.
            prepend (bool): Whether to prepend the edge to the front of the adjacent_from list.
        
        Raises:
            AssertionError: If the from_key or to_key is not found in the graph.
            AssertionError: If the edge already exists between the two nodes.
        
        Note:
            It is recommended to use temporary edges to represent only dynamic changes in the graph.
        """
        node_from = self.get_node_with_temporary(from_key)
        node_to = self.get_node_with_temporary(to_key)
        assert node_from is not None and node_to is not None, "Node ({}) not found in graph".format(from_key)
        assert not self.has_edge_with_temporary(from_key, to_key), "Edge ({}) already exists".format((from_key, to_key))
        node_from.adjacent_to.append(node_to)
        if prepend:
            node_to.adjacent_from.insert(0, node_from)
        else:
            node_to.adjacent_from.append(node_from)
        if (from_key, to_key) in self.temporary_removed_edges:
            self.temporary_removed_edges.remove((from_key, to_key))
        else:
            self.temporary_edges.append((from_key, to_key))

        assert to_key not in self.history.keys(), "Cannot add edge to a node ({}) that has already been evaluated".format(to_key)
        if from_key in self.history.keys() and to_key not in self.queue: # if from_key has been evaluated, add to queue
            self.queue.append(to_key)
        

    def remove_edge_temporary(self, from_key, to_key):
        """Remove a temporary edge between two nodes in the graph.

        This function temporarily removes a edge between the two nodes. Temporary
        removals are reverted after each evaluation.

        Args:
            from_key (str): The key of the node to remove the edge from.
            to_key (str): The key of the node to remove the edge to.

        Raises:
            AssertionError: If the edge does not exist between the two nodes.
            AssertionError: If the from_key or to_key is not found in the graph.
        
        Note:
            It is recommended to use this function only for dynamic changes in the graph.
        """
        assert self.has_edge_with_temporary(from_key, to_key), "Edge does not exist"
        node_from = self.get_node_with_temporary(from_key)
        node_to = self.get_node_with_temporary(to_key)
        node_from.adjacent_to.remove(node_to)
        node_to.adjacent_from.remove(node_from)
        if (from_key, to_key) in self.temporary_edges:
            self.temporary_edges.remove((from_key, to_key))
        else:
            self.temporary_removed_edges.append((from_key, to_key))

        assert to_key not in self.history.keys(), "Cannot remove edge to a node ({}) that has already been evaluated".format(to_key)
        assert from_key not in self.history.keys(), "Cannot remove edge from a node ({}) that has already been evaluated".format(from_key)

    def skip_nodes_temporary(self, keys):
        """Skip nodes temporarily in the graph.

        This function temporarily skips nodes in the graph.
        The previously recorded outputs of the skipped nodes
        will be reused in the evaluation. Temporary skips are
        reverted after each evaluation.

        Args:
            keys (list): A list of keys of the nodes to skip.
        
        Raises:
            AssertionError: If the key is not found in the graph.
        
        Note:
            It is recommended to use this function only for dynamic changes in the graph.
        """
        for key in keys:
            node = self.get_node_with_temporary(key)
            # assert node is not temporary
            assert key not in self.temporary_nodes.keys(), "Cannot skip temporary node: {}".format(key)
            assert key not in self.history.keys(), "Cannot skip node {}. It has already been evaluated.".format(key)
            node.skip_turn()

    def clean_temporary(self):
        for from_key, to_key in self.temporary_edges:
            node_from = self.get_node_with_temporary(from_key)
            node_to = self.get_node_with_temporary(to_key)
            node_from.adjacent_to.remove(node_to)
            node_to.adjacent_from.remove(node_from)
        for from_key, to_key in self.temporary_removed_edges:
            node_from = self.get_node_with_temporary(from_key)
            node_to = self.get_node_with_temporary(to_key)
            node_from.adjacent_to.append(node_to)
            node_to.adjacent_from.append(node_from)
        self.temporary_nodes = {}
        self.temporary_edges = []
        self.temporary_removed_edges = []
    
    def set_wandb_root_span(self, span):
        self.wandb_root_span = span
    
    def set_trace(self):
        if self.wandb_root_span is None:
            return
        chain_span = Trace(
            "GraphChain",
            kind="chain",
            start_time_ms=round(datetime.datetime.now().timestamp() * 1000),
            metadata={"num_iter": self.num_iter},
        )
        self.chain_span = chain_span

    def commit_trace(self):
        if self.chain_span is None:
            return
        self.chain_span._span.end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        self.chain_span.add_inputs_and_outputs(inputs={}, 
                                               outputs={"response": self.history})
        self.wandb_root_span.add_child(self.chain_span)
        self.chain_span = None
    
    def get_streaming_history(self):
        assert len(self.history) == 0, "Error: This function should only be called before evaluate() is called."
        return self.history

    def evaluate(self):
        """Evaluate the graph in a topological order.

        This function evaluates the graph in a topological order. The order of evaluation
        is determined by the dependencies between the nodes. The graph can also have temporary
        nodes and edges. Temporary nodes and edges are used to represent dynamic changes in the
        graph during evaluation. Temporary nodes and edges are cleared after each evaluation.

        Returns:
            dict: A dictionary of the results from the graph.

        Raises:
            AssertionError: If the temporary nodes are not cleared before calling evaluate().
        """
        
        def recalculate_in_degree(key):
            node = self.get_node_with_temporary(key)
            return len([n for n in node.get_dependencies_inc_order() if n.key not in self.history.keys()]) 
        
        def find_next_node(queue):
            for key in queue:
                degree = recalculate_in_degree(key)
                if degree == 0:
                    return key
            return None
        
        assert len(self.temporary_nodes) == 0, "Temporary nodes must be cleared before calling evaluate()"
        in_degree = {key: len(self.nodes[key].adjacent_from) for key in self.nodes.keys()}
        self.queue = [key for key in in_degree.keys() if in_degree[key] == 0]
        self.order = []
        self.set_trace()
        while True:
            node_key = find_next_node(self.queue)
            if node_key is None:
                break
            self.queue.remove(node_key)
            self.order.append(node_key)
            node = self.get_node_with_temporary(node_key)
            self.history[node_key] = node.evaluate() # this may change the graph
            for adjacent in node.adjacent_to:
                if adjacent.key not in self.history.keys() and adjacent.key not in self.queue:
                    self.queue.append(adjacent.key)
        self.num_iter += 1
        self.commit_trace()
        self.history_list.append(self.history.copy())
        self.history = {}
        self.order = []
        self.queue = []
        self.clean_temporary()
        return self.history_list[-1]