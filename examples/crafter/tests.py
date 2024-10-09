import json

prompts = [
    str(i) for i in range(11)
]

edges = {
    2: [1],
    3: [1, 2],
    4: [1, 3],
    5: [2, 4],
    6: [1, 4],
    7: [6],
    8: [6],
    9: [2, 6],
    10: [3, 4, 5, 7, 8, 9],
}

database = {
    "A": "a",
    "B": "b",
    "C": "c",
}

# level 1 feature:
# allow user to specify a query_model function for each prompt. 
# Run query_model on all '''prompts''' according to topological order defined by '''edges'''
# expected return value: a dictionary mapping prompt to result.

# level 2 feature:
# allow user to specify a compose_prompt function for each prompt.
# Run compose_prompt before query_model on each prompt.
#
# compose_prompt takes the following arguments:
# dependencies: list of (prompt, result) pairs. We want to query LLM for all the dependencies before querying LLM for the current prompt.
# prompt: the current prompt we want to query LLM for.
# database: a database that supports the following operations (can be accessed by the user as well):
#   database.get(key) -> value
#   database.put(key, value)
#   database.delete(key)
#   database.clear()
#   database.keys() -> list of keys
#   database.values() -> list of values
#   database.items() -> list of (key, value) pairs
#
# return value: a tuple (prompt, idx) directly passed to query_model.

# level 3 feature:
# add a history argument to compose_prompt.
#
# history stores a list of dictionary (see expected return value of level 1).
# history.clear() clears the history.
# adding to history should be automatic.
#
# This allows compose_prompt to access the results of previous queries.

# level 4 feature:
# allow user to specify a after_query function for each prompt.
# Run after_query after query_model on each prompt.
#
# after_query takes the following arguments:
# prompt: the current prompt we queried LLM for.
# result: the result returned by LLM.
# database: see level 2
# history: see level 3
#
# after_query can modify the database, and change how the graph is traversed.
# For example, one can add a new node to the graph, or remove nodes from the graph. (This should be made as easy as possible)
#
# return value: processed_result (can be None)
#
# Catch exceptions in after_query and re-attempt query_model with the exception message added to the prompt.
# If the re-attempt fails k times, return the exception message as the processed_result.


def compose_test_prompt(node):
    dependencies = node.get_dependencies()  # list of nodes
    prompt = ""
    for dep in dependencies:
        prompt += json.dumps(dep.represent()) + "\n"
    prompt += "Question: {}".format(node.get_prompt())
    return prompt


def query_model_test(msg, shrink_idx, max_gen=512, temp=0.):
    return "Answer to:\n\n{}".format(msg)