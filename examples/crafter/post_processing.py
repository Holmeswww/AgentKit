import json
from utils import parse_tuple
from agentkit import exceptions as ex
from agentkit import SimpleDBNode
from compose_prompt import ComposePlannerPrompt
from colorama import Fore
import traceback
from crafter_description import match_act
from agentkit import after_query as aq
import agentkit.utils as utils

class SubgoalAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ['subgoal', 'completion_criteria', 'guide']
        self.length = 3

    def post_process(self):
        parsed_answer = self.parse_json()
        self.node.db['subgoals']['subgoal'] = parsed_answer[-1]['subgoal']
        self.node.db['subgoals']['completion_criteria'] = parsed_answer[-1]['completion_criteria']
        self.node.db['subgoals']['guide'] = parsed_answer[-1]['guide']

class SkillAfterQuery(aq.BaseAfterQuery):

    def post_process(self):
        parsed_answer, error_msg = utils.extract_json_objects(self.node.result)

        error = None
        if parsed_answer is None:
            error = ex.AfterQueryError("Failed to parse answer", error_msg)
        elif parsed_answer[-1] is None or len(parsed_answer[-1])==0:
            error = ex.AfterQueryError("No answer", "Invalid Json: It seems that the last Json object in the output above is either invalid or empty.")
        elif type(parsed_answer[-1]) != dict:
            error = ex.AfterQueryError("Invalid answer", "Invalid Type: Expecting the last Json object to be dictionary, got length {} instead.".format(type(parsed_answer[-1])))
        elif len(parsed_answer[-1]) != 1:
            error = ex.AfterQueryError("Invalid answer", "Invalid Length: Expecting only one identified skill in the dictionary, got {} instead.".format(len(parsed_answer[-1])))
        elif list(parsed_answer[-1].values())[0] is None or len(list(parsed_answer[-1].values())[0])!=3:
            error = ex.AfterQueryError("Invalid answer", "Invalid Value: Expecting the value in the last Json dictionary to be `[description, supported parameters tuple, usage_guide]`, got length {} instead.".format(list(parsed_answer[-1].values())[0]))

        if error is not None:
            raise error
        
        skill_type = list(parsed_answer[-1].keys())[0]
        skill_desc, skill_param, skill_guide = parsed_answer[-1][skill_type]
        self.node.result_raw = self.node.result
        self.node.result = "[{},{},{}]".format(skill_type, skill_desc, skill_param, skill_guide)
        self.node.db['skills']['skill_library'][skill_type] = {
                'skill_desc': skill_desc,
                'skill_param': skill_param,
                'skill_guide': skill_guide,
            }
        self.node.db['skills']['skill'] = skill_type

class AdaptiveAfterQuery(aq.JsonAfterQuery):

    def post_process(self):

        if self.node.result.strip() == "N/A":
            self.node.result = "N/A"
            self.node.db['adaptive_questions'] = None
            return
        
        questions = """Answer the current questions based on the observation, gameplay history, knowledge base, and instruction manual.
In your answer, explicitly state 'missing' if something is missing from the instruction manual and the knowledge base. Do not make assumptions.

Questions:
{}""".format(self.node.result)

        self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db))
        for node in self.node.db['prompts']['adaptive_dependencies']:
            self.node.graph.add_edge_temporary(node, questions)

        for node in self.node.db['prompts']['adaptive_actor_questions']:
            self.node.graph.add_edge_temporary(questions, node, prepend=True)

        self.node.db['adaptive_questions'] = questions

class KBAddAfterQuery(aq.JsonAfterQuery):
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = []
        self.length = None

    def post_process(self):
        parsed_answer = self.parse_json()
        json_dict = parsed_answer[-1]
        new_knowledge = {}
        try:
            features = ['discovered', 'general', 'unknown', 'concrete_and_precise', 'solid']
            for k, v in json_dict.items():
                if False not in ['yes' in v[f].lower() for f in features]:
                    new_knowledge[k] = v['discovery_short']
        except Exception as e:
            raise ex.AfterQueryError("Invalid answer", "{}: {}".format(e, traceback.format_exc()))
        self.node.result_raw = self.node.result
        self.node.result = json.dumps(json_dict, sort_keys=True, indent=0)
        self.node.db['kb']['knowledge_base'].update(new_knowledge)


class KBReasonAfterQuery(aq.JsonAfterQuery):
        
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = []
        self.length = None

    def post_process(self):
        parsed_answer = self.parse_json()
        json_dict = parsed_answer[-1]
        unknowns = {}
        try:
            features = ['unknown', 'novel', 'general', 'relevant', 'correct']
            for k, v in json_dict.items():
                if False not in ['yes' in v[f].lower() for f in features]:
                    unknowns[k] = v['info']
        except Exception as e:
            raise ex.AfterQueryError("Invalid answer", "{}: {}".format(e, traceback.format_exc()))
        self.node.db['kb']['unknowns']={self.node.prompt: json.dumps(unknowns, sort_keys=True, indent=0)}
        self.node.db['kb']['unknowns_json']={self.node.prompt: unknowns}

class ReflectionAfterQuery(aq.JsonAfterQuery):
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = []
        self.length = 7

    def post_process(self):
        parsed_answer = self.parse_json()
        json_dict = parsed_answer[-1]
        if 'yes' in json_dict['unexpected_encounters'].lower():
            self.node.db["reflection"]["unexpected"].append(self.node.db["environment"]["step"])
        if 'yes' in json_dict['mistake'].lower():
            self.node.db["reflection"]["mistake"].append(self.node.db["environment"]["step"])
        if 'yes' in json_dict['correction_planned'].lower():
            self.node.db["reflection"]["correction"].append(self.node.db["environment"]["step"])
        if 'yes' in json_dict['confused'].lower():
            self.node.db["reflection"]["confusion"].append(self.node.db["environment"]["step"])
        if True in ['yes' in json_dict[k].lower() for k in ['unexpected_encounters', 'mistake', 'correction_planned', 'confused']]:
            self.node.db["reflection"]["all"].append(self.node.db["environment"]["step"])
        if True not in ['yes' in v.lower() for v in json_dict.values()] and len(self.node.db["history"]["qa_history"]) > 0:
            print(Fore.BLUE + "Skipping a bunch of reflection questions..." + Fore.RESET)
            self.node.graph.skip_nodes_temporary(self.node.db["prompts"]["reflection_skip_questions"])

class ListActionAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = []
        self.length = None

    def post_process(self):
        parsed_answer = self.parse_json()
        filtered_result = {}
        try:
            self.node.db['allowed_actions'] = []
            keys_to_keep_yes = ['target', 'allowed', 'unlock new achievement']
            keys_to_keep_no = ['target', 'allowed', 'reasoning']
            for action,v in parsed_answer[-1].items():
                if action.strip().lower() == "noop": # Skip noop. This doesn't change the behavior of the LLM experimentally but saves quite a bit of tokens.
                    continue
                if "yes" in v['allowed'].lower():
                    self.node.db['allowed_actions'].append(action)
                    filtered_result[action] = {k:v[k] for k in keys_to_keep_yes}
                else:
                    filtered_result[action] = {k:v[k] for k in keys_to_keep_no}
        except Exception as e:
            raise ex.AfterQueryError("Invalid answer", "{}: {}".format(e, traceback.format_exc()))
        self.node.result = json.dumps({k:str(i) for i,k in enumerate(filtered_result.keys())}, indent=0).strip()
        for i, v in enumerate(filtered_result.values()):
            self.node.result = self.node.result.replace('"{}"'.format(i), json.dumps(v))
        
        # Adaptive Questions
        if 'adaptive_questions' not in self.node.db or self.node.db['adaptive_questions'] is None:
            return

        questions = self.node.db['adaptive_questions']
        
        self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db, verbose=self.node.verbose))
        for node in self.node.db['prompts']['adaptive_dependencies']:
            self.node.graph.add_edge_temporary(node, questions)
        
        for node in self.node.db['prompts']['adaptive_strategy_questions']:
            self.node.graph.add_edge_temporary(questions, node, prepend=True)

class ActionSummaryAfterQuery(aq.JsonAfterQuery):
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ['plan-sketch', 'details', 'target', 'relevance-criteria', 'expiration-condition', 'notes']
        # self.length = 6

    def post_process(self):
        parsed_answer = self.parse_json()
        self.node.db['action_summary'] = parsed_answer[-1]
        self.node.db['action_notes'] = parsed_answer[-1]['notes']
        del self.node.db['action_summary']['notes']

class ActionAfterQuery(aq.JsonAfterQuery):
        
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ['action', 'repeats', 'hazards', 'obstacles']
        # self.length = 3

    def post_process(self):
        parsed_answer = self.parse_json()
        act, action_name, error_msg = match_act(parsed_answer[-1]['action'].replace("(","").replace("_"," "))
        if act is None:
            raise ex.AfterQueryError("Invalid answer", "Invalid action: {}".format(parsed_answer[-1]['action'], error_msg))
        if type(parsed_answer[-1]['repeats']) == str and not (parsed_answer[-1]['repeats']).isnumeric():
            raise ex.AfterQueryError("Invalid answer", "Invalid repeats: '{}'. Expecting an integer.".format(parsed_answer[-1]['repeats']))
        self.node.db['action'] = act
        if "move" in action_name.lower() and "yes" not in parsed_answer[-1]['hazards'].lower()and "yes" not in parsed_answer[-1]['obstacles'].lower():
            self.node.db['action_repeats'] = min(4,int(parsed_answer[-1]['repeats']))
        elif "do" in action_name.lower():
            self.node.db['action_repeats'] = min(3,int(parsed_answer[-1]['repeats']))
        else:
            self.node.db['action_repeats'] = 1

        self.node.result = json.dumps({
            'action': action_name,
            'repeats': self.node.db['action_repeats']
        }, indent=2)

class SummaryAfterQuery(aq.JsonAfterQuery):
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ['action', 'repeats', 'target', 'success', 'causes_of_failure']
        # self.length = 5

    def post_process(self):
        parsed_answer = self.parse_json()

        action_desc = ""
        if 'move' in parsed_answer[-1]['action'].lower():
            action_desc += "{}, {} steps".format(parsed_answer[-1]['action'], parsed_answer[-1]['repeats'])
        else:
            action_desc += "{}, {} steps, target: {}".format(parsed_answer[-1]['action'], parsed_answer[-1]['repeats'], parsed_answer[-1]['target'])
        if 'no' in parsed_answer[-1]['success'].lower():
            action_desc += " (failed, causes of failure: {})".format(parsed_answer[-1]['causes_of_failure'])
        else:
            action_desc += " (succeeded)"
        
        self.node.result = action_desc