from .exceptions import AfterQueryError
from .node_functions import error_msg_default
from collections.abc import Callable, Awaitable
from .graph import Graph
from .after_query import BaseAfterQuery
from .compose_prompt import BaseComposePrompt
from colorama import Fore, Back, Style
import copy
import datetime
try:
    from wandb.sdk.data_types.trace_tree import Trace
except:
    pass

class BaseNode:
    """Base class for a node in the graph.

    Each node in the graph is an instance of the BaseNode class. The node is evaluated by querying the LLM with a prompt.

    Attributes:
        key (str): Unique key for the node.
        prompt (str): Prompt for the node.
        result (str): Result of the node evaluation.
        temporary_skip (bool): Flag to skip the node evaluation.
        graph (Graph): Graph object.
        adjacent_to (list): List of nodes that are adjacent to this node.
        adjacent_from (list): List of nodes that are adjacent from this node.
        evaluate_after (list): List of nodes that are evaluated after this node.
        counts (list): List of token counts.
        query_llm (Callable): Function to query the LLM.
        _compose_prompt (BaseComposePrompt): ComposePrompt object.
        after_query (BaseAfterQuery): AfterQuery object.
        error_msg_fn (Callable): Function to add error message to the prompt.
        verbose (bool): Verbose flag.
        token_counter (Callable): Function to count tokens.
    """
    def __init__(self, key:str, prompt:str, graph:Graph, query_llm:Callable, compose_prompt:BaseComposePrompt, after_query:BaseAfterQuery=None, error_msg_fn:Callable[[list,str,AfterQueryError],list]=error_msg_default, verbose:bool=False, token_counter:Callable=None):
        """Initializes the BaseNode class.
        
        Args:
            key (str): Unique key for the node.
            prompt (str): Prompt for the node.
            graph (Graph): Graph object.
            query_llm (Callable): Function to query the LLM.
            compose_prompt (BaseComposePrompt): ComposePrompt object.
            after_query (BaseAfterQuery): AfterQuery object.
            error_msg_fn (Callable): Function to add error message to the prompt.
            verbose (bool): Verbose flag.
            token_counter (Callable): Function to count tokens.
        """
        self.key = key
        self.prompt = prompt
        self.result = None
        self.temporary_skip = False
        self.temporary_skip = False
        self.graph = graph
        self.adjacent_to = []  # this -> node
        self.adjacent_from = []  # node -> this
        self.evaluate_after = []  # node -> this
        self.counts = []
        self.query_llm = query_llm
        self._compose_prompt = compose_prompt
        self.after_query = None
        self.chain_span = None
        if after_query is not None:
            self.after_query = after_query
            self.after_query.set_node(self)
        self.verbose = verbose
        self._add_error_msg = error_msg_fn
        
        if token_counter is not None:
            self.token_counter = token_counter
        else:
            self.token_counter = None
    
    def get_dependencies(self):
        """Get the dependencies (edges) of the node.

        Returns:
            list: List of dependencies.
        """
        return self.adjacent_from
    
    def get_dependencies_inc_order(self):
        """Get the dependencies (edges) and order (non-dependency edges) of the node.
        
        Returns:
            list: List of dependencies.
        """
        return self.adjacent_from + self.evaluate_after

    def compose_prompt(self):
        return self._compose_prompt(dependencies=self.get_dependencies(), prompt=self.prompt)

    def _print_question(self):
        if self.verbose:
            print(Fore.CYAN + "Prompt: " + Style.DIM + "{}".format(self.prompt) + Style.RESET_ALL)
    
    def _print_answer(self, msg):
        if self.verbose:
            print("Answer: " + Style.DIM + "{}".format(msg) + Style.RESET_ALL)
    
    def _print_skip(self):
        if self.verbose:
            print("Skip: " + Style.DIM + "{}".format(self.prompt) + Style.RESET_ALL)
    
    def skip_turn(self):
        """Temporarily skip the node evaluation. The previous result of the node will be reused.

        Raises:
            AssertionError: If the node has never been evaluated.
        """
        assert self.result is not None, "Attempting to skip a node ({}) that has never been evaluated".format(self.key)
        self.temporary_skip = True
    
    def _query_llm(self, prompt, shrink_idx):

        start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        result = self.query_llm(prompt, shrink_idx)
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)

        if self.chain_span is not None:
            llm_span = Trace(
                "OpenAI",
                kind="llm",
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                inputs={"prompt":prompt},
                status_code="success",
                outputs={"response": result},
            )
            self.chain_span.add_child(llm_span)
        return result
    
    def _after_query(self, ignore_errors=False):
        if self.after_query is None:
            return
        status_code = "success"
        status_message = ""
        error = None
        after_query_input = self.result
        try:
            start_time_ms = round(datetime.datetime.now().timestamp() * 1000)
            self.after_query()
        except AfterQueryError as e:
            status_code = "error"
            status_message = e.error
            error = e
            self.result = after_query_input
        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        if self.chain_span is not None:
            tool_span = Trace(
                "AfterQuery",
                kind="tool",
                status_code=status_code,
                status_message=status_message,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                inputs={"input": after_query_input},
                outputs={"response": self.result},
            )
            self.chain_span.add_child(tool_span)
        if status_code == "error":
            if ignore_errors:
                print(Fore.RED + "WARNING: ({}, {}) for {}".format(error, status_message, self.result) + Fore.RESET)
                self.result = "N/A"
            else:
                raise error

    def set_trace(self):
        if self.graph.chain_span is None:
            self.chain_span = None
            return
        chain_span = Trace(
            "NodeChain",
            kind="chain",
            start_time_ms=round(datetime.datetime.now().timestamp() * 1000),
            metadata={"prompt": self.prompt, "key": self.key},
        )
        self.chain_span = chain_span

    def commit_trace(self):
        if self.chain_span is None:
            return
        self.chain_span._span.end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        self.chain_span.add_inputs_and_outputs(inputs={"dependencies": [(node.key,node.result) for node in self.adjacent_from]}, 
                                               outputs={"response": self.result})
        self.graph.chain_span.add_child(self.chain_span)
        self.chain_span = None

    def evaluate(self):
        """Evaluate the node by querying the LLM.

        Retries the AfterQuery with the LLM up to 3 times in case of an error.

        Returns:
            str: Result of the node evaluation.

        Raises:
            AssertionError: If any dependency of the node has not been evaluated.
        """
        for node in self.adjacent_from:
            assert node.result is not None, "Dependency {} of {} has been not evaluated".format(node.key, self.key)

        if not self.temporary_skip:
            self.set_trace()
            prompt, shrink_idx = self.compose_prompt()
            error = None
            self._print_question()
            for i in range(3):
                try:
                    temp_prompt = copy.deepcopy(prompt)
                    if error is not None:
                        temp_prompt = self._add_error_msg(temp_prompt, self.result, error)
                    self.result, usage = self._query_llm(temp_prompt, shrink_idx)
                    if usage is not None:
                        self.counts.append(copy.copy(usage))
                    elif self.token_counter is not None:
                        self.counts.append({'prompt': self.token_counter(temp_prompt), 'completion':self.token_counter(self.result)})
                    self._after_query(ignore_errors=(i==2))
                    break
                except AfterQueryError as e:
                    error = e.error
            self.commit_trace()
            self._print_answer(self.result)
            print()
        else:
            self.temporary_skip = False
        return self.result

    def get_token_counts(self):
        """Get the LLM token counts for the specific node since instantiation.

        Returns:
            dict: Token counts for the node.
        """

        if self.token_counter is None:
            print("Warning: token counter not set for node {}".format(self.key))
        return {
            'calls': len(self.counts),
            'prompt': sum([c['prompt'] for c in self.counts]),
            'completion': sum([c['completion'] for c in self.counts])
        }