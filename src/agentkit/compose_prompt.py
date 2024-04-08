import re
class BaseComposePrompt:
    """Base class for composing prompts. This class is used to compose prompts for the LLM.

    Each compose prompt instance takes dependencies (QA results from dependent nodes) and 
    a prompt as input and returns a prompt in OpenAI format.

    Attributes:
        system_prompt (str): System prompt to be used in the conversation.
    """

    def __init__(self, system_prompt:str = "You are a helpful assistant."):
        """Initializes the BaseComposePrompt class.

        Args:
            system_prompt (str): System prompt to be used in the conversation.
        """
        self.system_prompt = system_prompt
    
    def compose(self, dependencies, prompt):
        """Compose prompt for the LLM.
        
        Args:
            dependencies (list): List of dependencies.
            prompt (str): User prompt.
        
        Returns:
            msg (list): List of messages in OpenAI format.
            shrink_idx (int): Index to shrink the prompt in case of truncation.
        """
        msg = [{"role": "system", "content": self.system_prompt}]
        msg = self.add_dependencies(msg, dependencies)
        msg.append({"role": "user", "content": prompt})
        return msg, 1

    def __call__(self, dependencies, prompt):
        return self.compose(dependencies, prompt)

    def add_dependencies(self, messages, dependencies):
        """Add dependencies to the messages.

        Args:
            messages (list): List of messages.
            dependencies (list): List of dependencies.
        
        Returns:
            messages (list): List of messages with dependencies in OpenAI format.
        """
        if len(dependencies)>0:
            for node in dependencies:
                messages.append({"role": "user", "content": node.prompt})
                messages.append({"role": "assistant", "content": node.result})
                assert type(messages[-1]['content']) == str, "Invalid type: {}".format(type(messages[-1]['content']))
        return messages

class ComposePromptDB(BaseComposePrompt):
    """Class for composing prompts with database values. This class is used to compose prompts for the LLM.

    Each compose prompt instance takes dependencies (QA results from dependent nodes) and
    a prompt as input and returns a prompt in OpenAI format.

    Attributes:
        system_prompt (str): System prompt to be used in the conversation.
        node (Node): Corresponding Node object.
    """
    
    def __init__(self, system_prompt:str = "You are a helpful assistant."):
        """Initializes the ComposePromptDB class.

        Args:
            system_prompt (str): System prompt to be used in the conversation.
        """
        super().__init__(system_prompt)
        self.node = None
    
    def set_node(self, node):
        """Set the node for the ComposePromptDB class.

        Args:
            node (Node): Corresponding Node object.
        """
        self.node = node
    
    def compose(self, dependencies, prompt):
        """Compose prompt for the LLM with database augmentation.

        Prompt text may contain placeholders for db values. This function will replace the placeholders with the actual values from db.

        Args:
            dependencies (list): List of dependencies.
            prompt (str): User prompt.
        
        Returns:
            msg (list): List of messages in OpenAI format.
            shrink_idx (int): Index to shrink the prompt in case of truncation.
        """

        msg = [{"role": "system", "content": self.system_prompt}]

        msg = self.add_dependencies(msg, dependencies, self.node.db)

        prompt, db_retrieval_results = self.render_db(prompt, self.node.db)
        self.node.rendered_prompt = prompt
        self.node.db_retrieval_results = db_retrieval_results

        msg.append({"role": "user", "content": prompt})

        return msg, self.shrink_idx


    def __call__(self, dependencies, prompt):

        if self.node is None:
            raise Exception("Node is not set")
        
        return self.compose(dependencies, prompt)
    
    def render_db(self, text, db):
        """Render db placeholders in the prompt text.

        Prompt text may contain placeholders for db values. This function will replace the placeholders with the actual values from db.

        Example:

        1. Confirm if the subgoal '$db.subgoals.subgoal$' is still accuracte for the challenge.

        2. Confirm if the subgoal '$db.subgoals.subgoal$' is incomplete and up-to-date according to the completion criteria '$db.subgoals.guide$'.
        
        Args:
            text (str): Prompt text.
            db (dict): Database values.
        
        Returns:
            text (str): Prompt text with db placeholders replaced with actual values.
            db_retrieval_results (list): List of db retrieval results.
        """
        db_retrieval_results = []
        pattern = r'\$db\.(.*?)\$'
        matches = re.findall(pattern, text)
        for match in matches:
            value = db
            for key in match.split('.'):
                if key in value:
                    value = value[key]
                else:
                    value = None
                    break
            if value != "None":
                text = text.replace(f'$db.{match}$', str(value))
            else:
                text = text.replace(f'$db.{match}$', "'NA'")
            db_retrieval_results.append((match, value))

        return text, db_retrieval_results

    def add_dependencies(self, messages, dependencies, db):
        """Add dependencies to the messages with db values.

        Args:
            messages (list): List of messages.
            dependencies (list): List of dependencies.
            db (dict): Database values.
        
        Returns:
            messages (list): List of messages with dependencies in OpenAI format.
        """
        if len(dependencies)>0:
            for node in dependencies:
                if node.key in db['shorthands'].keys():
                    messages.append({"role": "system", "content": "{}:\n\n{}".format(db['shorthands'][node.key], node.result)})
                else:
                    assert node.rendered_prompt != None, "Rendered prompt is not set for node {}".format(node.key)
                    messages.append({"role": "user", "content": node.rendered_prompt})
                    messages.append({"role": "assistant", "content": node.result})
                assert type(messages[-1]['content']) == str, "Invalid type: {}".format(type(messages[-1]['content']))
        return messages