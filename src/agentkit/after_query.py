from .utils import extract_json_objects
from .exceptions import AfterQueryError

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

class JsonAfterQuery(BaseAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.values = []
        self.length = None

    def parse_json(self):
        parsed_answer, error_msg = extract_json_objects(self.node.result)
        if parsed_answer is None:
            raise AfterQueryError("Failed to parse answer", error_msg)
        elif parsed_answer[-1] is None:
            raise AfterQueryError("No answer", "Invalid Json: It seems that the last Json object in the output above is invalid.")
        elif type(parsed_answer[-1]) != self.type:
            raise AfterQueryError("Invalid answer", "Invalid Type: Expecting the last Json object to be {}, got {} instead.".format(self.type, type(parsed_answer[-1])))
        if self.length is not None and len(parsed_answer[-1]) != self.length:
            raise AfterQueryError("Invalid answer", "Expecting length {}, got {} instead.".format(self.length, len(parsed_answer[-1])))

        for k in self.values:
            if k not in parsed_answer[-1].keys():
                raise AfterQueryError("Invalid answer", "Expecting '{}' in the keys.".format(k))
        
        return parsed_answer