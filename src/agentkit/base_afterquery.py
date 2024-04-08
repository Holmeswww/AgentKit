class BaseAfterQuery:
    """Base class for after query postprocessing.

    Each after query instance performs postprocessing after the LLM query.

    Attributes:
        node (BaseNode): Node object.
    """

    def __init__(self):
        self.node = None
    
    def set_node(self, node):
        """Set the node for the after query.

        Args:
            node (BaseNode): Node object.
        """
        self.node = node
        return self
    
    def post_process(self):
        """Post process the result of the LLM query.
        
        This method can be overridden by the derived class to perform postprocessing.
        """
        pass

    def __call__(self):
        if self.node is None:
            raise Exception("Node is not set")

        self.post_process()