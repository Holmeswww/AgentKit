from .exceptions import AfterQueryError
from collections.abc import Callable, Awaitable
from .node_functions import error_msg_default
from .base_node import BaseNode
from .graph import Graph
from colorama import Fore, Back, Style
import typing as t

from .after_query import BaseAfterQuery
from .compose_prompt import ComposePromptDB

class SimpleDBNode(BaseNode):
    """Class for a node in the graph that queries a database.

    Each node in the graph is an instance of the SimpleDBNode class. The node is evaluated by querying the LLM with a prompt.

    Attributes:
        db (Any): Database object.
    """
    def __init__(self, key:str, prompt: str, graph: Graph, query_llm: Callable, compose_prompt: ComposePromptDB, database: t.Any, after_query: BaseAfterQuery = None, error_msg_fn: Callable[[list, str, AfterQueryError], list] = error_msg_default, verbose: bool = False, token_counter: Callable = None):
        """Initializes the SimpleDBNode class.

        Args:
            key (str): Unique key for the node.
            prompt (str): Prompt for the node.
            graph (Graph): Graph object.
            query_llm (Callable): Function to query the LLM.
            compose_prompt (ComposePromptDB): ComposePromptDB object.
            database (Any): Database object.
            after_query (BaseAfterQuery): AfterQuery object.
            error_msg_fn (Callable): Function to add error message to the prompt.
            verbose (bool): Verbose flag.
            token_counter (Callable): Function to count tokens.
        """
        super().__init__(key, prompt, graph, query_llm, compose_prompt, after_query=after_query, error_msg_fn=error_msg_fn, verbose=verbose, token_counter=token_counter)
        self.db: t.Any = database
        self._compose_prompt.set_node(self)
        self.rendered_prompt = None
    
    def compose_prompt(self):
        return self._compose_prompt(dependencies=self.get_dependencies(), prompt=self.prompt)

    def _print_answer(self, msg):
        if self.verbose:
            if hasattr(self, 'db_retrieval_results') and len(self.db_retrieval_results) > 0:
                print("DB operations: " + Style.DIM + Fore.BLUE + "{}".format(self.db_retrieval_results) + Style.RESET_ALL)
            print("Answer: " + Style.DIM + "{}".format(msg) + Style.RESET_ALL)