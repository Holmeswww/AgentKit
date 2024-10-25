from agentkit import Graph, SimpleDBNode
from compose_prompt import *
from post_processing import *

query_fast = 'query_fast'
query_fast_accurate = 'query_fast_accurate'
query_reason = 'query_reason'
query_plan_accurate = 'query_plan_accurate'
query_spatial = 'query_spatial'

prompts = {

'obs_obj':{
'prompt':"""
First, describe all objects the player faces or around the player. Describe the object type, the direction, the distance, the coordinates, and the requirements to interact with the object from the instruction manual.
Be precise and accurate with the direction, coordinates.
Output a Json list in the following format:
[
{"object":$type, "direction":$precise_direction, "distance":$distance$, "facing":$[yes/no], 'coordinate':"precise coordinates", "requirements":"requirements to interact from instruction manual, put 'NA' if not applicable"},
...
]

Second, in one sentence, describe what the player sees further to the front.
""",
'dep':[],
'shorthand': "Current observation/surroundings",
'compose': ComposeObservationPrompt(),
'query': query_fast_accurate,
},

'obs_inv':{
'prompt':"""Describe the player's inventory as a json dictionary, with the item_names as key and item_amount as value. Write '{}' if inventory is empty.""",
'dep':[],
'shorthand': "Current inventory",
'compose': ComposeObservationPrompt(),
'query': query_fast,
},

'obs_vit':{
'prompt':"""Describe the player's vitals as a json dictionary, in the format of {vital_name: "value/max"}.""",
'dep':[],
'shorthand': "Current vitals",
'compose': ComposeObservationPrompt(),
'query': query_fast,
},

'obs_chg':{
'prompt':"""Using a list, document any changes to the player observation, inventory, and status in the most recent step.
Be precise about changes to numbers and details about the vitals, inventory, and the distance to objects around the player.
Be concise and only list the changes. Output 'NA' if there is no change""",
'dep':[],
'shorthand': "Changes to the observation in the current step",
'compose': ComposeObservationReasoningPrompt(),
'query': query_reason,
},

'obs_last_act':{
'prompt':"""First, consider how the most recent action should have changed the player's observation/status/inventory, be specific with the action and the target.
Then, deduce if the last action succeeded.
If yes, end the answer here.
If not, identify potential cause for the failure, by focusing on the player's surrounding/observation, inventory, instruction manual, and missing information.
Be concise with the language, but precise with the details.""",
'dep':['obs_chg',],
'compose': ComposeActionReflectionPrompt(),
'query': query_spatial,
},


's-obs-obs':{
'prompt':"""In two sentences, describe the observation and surroundings, including all object types and what the player is facing.
Make sure to include the location details of objects relevant to the current plan '$db.action_summary.plan-sketch$' and target '$db.action_summary.target$'.""",
'dep':['obs_obj'],
'shorthand': "Player observation",
'compose': ComposeSummaryPrompt(),
'query': query_fast,
},

's-obs-vitals':{
'prompt':"In one sentence, describe all inventory items and the current vitals.",
'dep':['obs_inv', 'obs_vit'],
'shorthand': "Player vitals",
'compose': ComposeSummaryPrompt(),
'query': query_fast,
},

's-action':{
'prompt':"""Output a Json dictionary of the following format:
```
{
"action": $action, # The most recent action, including the direction if it's a movement action
"repeats": $repeats # The number of times the action was repeated
"target": $target # The target of the action. Write 'NA' if the action is a movement/sleep action.
"success": $success # [yes/no] If the action succeeded
"causes_of_failure": $causes_of_failure # In one sentence, write the cause(s) of the failure. Write 'NA' if the action succeeded.
}
```
""",
'dep':['obs_last_act',],
'shorthand': "Most recent action and result",
'compose': ComposeSummaryPrompt(),
'query': query_fast,
'after_query': SummaryAfterQuery(),
},

'obs_current_actions':{
'prompt':"""For each action in the numbered list of all actions, reason with the current observation/surroundings and identify if the action is allowed at the current gamestep according to the requirements.
Then predict the target the action will interact with.
Format the output as a Json dictionary of the following format:
{
"$action": {"requirements": $requirements of the action, "related observation": $what in the observation may affect/block the action?, "reasoning": $precise but very concise reason of why the action is blocked, "allowed": $yes/no, "target": $inferred target, "unlock new achievement": $yes/no, will the success of this action unlock an unaccomplished achievement?}, # do not output white spaces
...
}
""",
'dep':['s-action', 'obs_obj'],
'shorthand': "List of all actions",
'compose': ComposeObservationActionReasoningPrompt(),
'query': query_plan_accurate,
'after_query': ListActionAfterQuery(),
},

'reflect':{
'prompt':"""Consider how the player reached the current state, concisely summarize the player's high-level trajectory in the gameplay history. Focus on the present and the past and do not make future predictions.
Has the player been making effective progress towards the top subgoal '$db.subgoals.subgoal$'? Has the player been effectively addressing all the mistakes/obstacles encountered?
Output 'NA' if there is no history.""",
'dep': ['obs_obj', 'obs_inv', 'obs_vit', 'obs_chg', 'obs_last_act'],
'after': ['s-obs-obs', 's-obs-vitals', 's-action',],
'shorthand': "Gameplay history",
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'planner_unexpected':{
'prompt':"""Did the player encounter any obstacles or unexpected scenarios? Also, did player status change unexpectedly?
Write \"NA\" and end answer here if no expectancy.
Otherwise, should the player modify the subgoals to react?""",
'dep':['s-obs-obs', 's-obs-vitals', 'obs_obj', 'obs_inv', 'obs_chg', 'obs_last_act'],
'after': ['s-action',],
'short': "Past Unexpected Scenarios",
'compose': ComposePlannerReflectionPrompt(),
'query': query_reason,
},

'planner_mistake':{
'prompt':"""Did the player make any mistakes approaching the top subgoal '$db.subgoals.subgoal$'? Write \"NA\" and stop the answer here if there's no mistake.
Otherwise, identify the potential causes of the mistake and find the most probable cause by analyzing the current situation.
Then, suggest a precise and executable modification to the subgoals to address the mistake.""",
'dep':['s-obs-obs', 's-obs-vitals', 'obs_chg', 'obs_current_actions'],
'after': ['s-action',],
'short': "Past Mistakes",
'compose': ComposePlannerReflectionPrompt(),
'query': query_reason,
},

'challenge':{
'prompt':"""Identify three high level challenges.
Based on the current situation, score their urgency out of 5 (5 is the highest urgency) and score their convenience out of 5 (5 being most convenient).
Then choose a challenge to address based on the urgency, convenience, and overall objective of the game.
Be concise.""",
'dep':['reflect', 'planner_mistake', 'obs_obj', 'obs_inv', 'obs_vit'],
'after': ['s-action',],
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'achievements':{
'prompt':"""Reason with the current situation and the requirements of the *current unaccomplished achievements*.
Identify a list of unaccomplished achievements that are easily unlockable and relevant to the current situation.
For each achievement, write a one-sentence explanation of how the achievement may be easily unlocked, and how it may benefit.""",
'dep':['reflect', 'obs_obj', 'obs_inv', 'obs_vit', 'obs_current_actions'],
'shorthand': "Relevant unaccomplished achievements",
'after': ['s-action',],
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'gate-plan_sketch':{
'prompt':"""Reason with the instruction manual and knowledge base. Explain concisely how the chosen challenge may be approached given the player's current situation, with actions permitted by the game.
Then, 
1. Confirm if the subgoal '$db.subgoals.subgoal$' is still accurate for the chosen challenge.
2. Confirm if the subgoal '$db.subgoals.subgoal$' is incomplete and up-to-date according to the completion criteria '$db.subgoals.completion_criteria$'.

If yes to both, end the answer here.
Then, sketch an updated high-level plan for addressing the chosen challenge. Do not plan to go back to something if you cannot provide it's location.
If situation allows, the plan-sketch should aim for achieving relevant unaccomplished achievements in the order of difficulty, without impacting player safety.""",
'dep':['reflect', 'planner_unexpected', 'planner_mistake', 'obs_obj', 'obs_inv', 'obs_vit', 'obs_current_actions', 'achievements', 'challenge'],
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'gate':{
'prompt':"""Reason with the previous conversation and context, and output a Json dictionary with answers to the following questions:
```
{
"unexpected_encounters": $ANSWER, # [yes/no] Did the player encounter any unexpected scenarios or obstacles?
"mistake": $ANSWER, # [yes/no] Did the player make any mistakes?
"correction_planned": $ANSWER, # [yes/no] Was correction planned at current step for the mistake?
"confused": $ANSWER, # [yes/no] Does the player seem confused?
"top_subgoal_completed": $ANSWER, # [yes/no] Is the most recent top subgoal complete according to the completion criteria?
"top_subgoal_changed": $ANSWER, # [yes/no] Has the most recent top subgoal been changed?
"replan": $ANSWER # [yes/no] Was there a re-plan/change for the plan sketch?
}
```
""",
'dep':['reflect', 'planner_unexpected', 'planner_mistake', 'gate-plan_sketch'],
'compose': ComposePlannerPrompt(),
'query': query_reason,
'after_query': ReflectionAfterQuery(),
},

'subgoals':{
'prompt':"""List the top 3 subgoals for the player according to the plan sketch. The subgoals should be quantifiable and actionable with in-game actions. Put subgoals of highest priority first.
For each subgoal, specify how the completion may be precisely quantified in at most one sentence (include the numbers and details).
Do not include additional information other than the subgoals and quantification.""",
'dep':['reflect', 'obs_obj', 'obs_inv', 'obs_vit', 'gate-plan_sketch'],
'after': ['gate'],
'shorthand': "Current subgoals (decreasing priority)",
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'top-subgoal':{
'prompt':"""Write the highest priority subgoal and completion criteria as a Json dictionary of the following format:
```
{
"subgoal": $subgoal, # The highest priority subgoal
"completion_criteria": $completion_criteria # The completion criteria for the subgoal
"guide": $guide # A brief high-level guide for completing the subgoal
}
```
""",
'dep':['reflect', 's-action', 'obs_current_actions', 'subgoals'],
'shorthand': "Current top subgoal and completion criteria",
'compose': ComposePlannerPrompt(),
'query': query_reason,
'after_query': SubgoalAfterQuery(),
},

'subgoal_analysis':{
'prompt':"""Check the instruction manual for requirements to complete the top subgoal.
Identify a list of unknown/missing information/details and do not make any assumptions.
Pay close attention to the details such as the numbers and specifications.""",
'dep':['reflect', 'obs_current_actions', 'top-subgoal'],
'shorthand': "Requirement and Missing Info Analysis for the Top Subgoal",
'compose': ComposePlannerPrompt(),
'query': query_reason,
},

'skill':{
'prompt':"""First, try to find an existing skill from the 'skill library' that could be used to represent the current top subgoal.

If none of the existing skills could be applied, create a new skill.

The skill, parameters, and guide should be general enough to be reusable for tasks of the same class.

Format the chosen skill as a Json object of the following format:
```
{$skill_name: [$1_line_skill_desciption, $supported_parameters, $skill_guide]}
```
""",
'dep':['top-subgoal'],
'compose': ComposeSkillPrompt(),
'query': query_reason,
'after_query': SkillAfterQuery(),
},

'planner-adaptive':{
'prompt':"""List at most 3 questions to help guide the agent on the subgoal: '$db.subgoals.subgoal$'. The questions should prompt the player to think concretely and precisely.
Example questions could ask the player to:
 - identify goal-relevant details from the observation, history, and instruction manual
 - recall related historical encounters
 - potential obstacles and how to overcome them
 - specify concrete ways to improve efficiency

Output only the list of questions and nothing else. Write \"NA\" if there's no question to ask.""",
'dep':['reflect', 'planner_unexpected', 'planner_mistake', 'top-subgoal', 'subgoal_analysis'],
'compose': ComposePlannerPrompt(),
'query': query_reason,
'after_query': AdaptiveAfterQuery(),
},

'kb-add':{
'prompt':"""Rewrite the unknown information and details list into a Json dictionary.
Give each item a concise but precise name as the key, and a dictionary of answers to the following inquiries as the value:
```
"item_name":{
"discovered": $ANSWER, # Can this unknown information be precisely uncovered or determined at the current gamestep? [yes/no] 
"discovery": $ANSWER, # In concise but precise terms, write what has been uncovered [If the information is not uncovered, write 'NA'.]
"discovery_short": $ANSWER, # Condensed version of 'discovery', only containing precisely the discovered info [If the information is not uncovered, write "NA".]
"general": $ANSWER, # Confirm that this uncovered information remain unchanged in subsequent steps of this game. [yes/no]
"unknown": $ANSWER, # Confirm that this uncovered information is missing from instruction manual. [yes/no]
"concrete_and_precise": $ANSWER, # Is the uncovered information concrete and precise enough to add to instruction manual? [yes/no]
"solid": $ANSWER, # Confirm that this information is not speculative or deduced based on assumption. [yes/no]
}
```

Include answers with a one-sentence justification or detail. Do not repeat the inquiries.
Write '{}' if there is nothing in the list of 'unknown information and details'.""",
'dep':['planner_unexpected', 'planner_mistake', 'top-subgoal', 'subgoal_analysis', 'obs_obj', 'obs_inv', 'obs_vit', 'obs_chg', 'obs_last_act'],
'shorthand': 'discovered information and details',
'compose': ComposeKBAddPrompt(),
'query': query_reason,
'after_query': KBAddAfterQuery(),
},

'unknown':{
'prompt':"""Merge the 'unknown/missing information' in the previous answer and the 'Previous unknown information and details' into a single Json dictionary.
Give each item a concise but precise name as the key, and a dictionary of answers to the following inquiries as the value:
```
"item_name": { # if applicable, use the same item name as in the previous answer or knowledge base
"info": $ANSWER, # In concise but precise language, describe what exactly is missing. [If the information is not missing, write 'NA'.]
"knowledge": $ANSWER, # What do you already know about the current requested info? [If nothing is known, write 'NA'.]
"unknown": $ANSWER, # Confirm that this requested info is missing from the instruction manual. [yes/no]
"novel": $ANSWER, # Confirm that the knowledge base does not already contain precise answer to this requested info. [yes/no]
"general": $ANSWER, # Is the requested info expected to remain unchanged in future game steps? [yes/no]
"relevant": $ANSWER, # Is this requested info helpful for succeeding the game? [yes/no]
"correct": $ANSWER # Confirm that this request does not disagree with the instruction manual. [yes/no] followed by a one-sentence justification.
}
```
Only include the answers, not the inquiries. Remove duplicates and arrange multi-target inquiries into separate items.""",
'dep':['subgoal_analysis'],
'after': ['kb-add'],
'shorthand': 'Unknown information and details',
'compose': ComposeKBReasonPrompt(),
'query': query_reason,
'after_query': KBReasonAfterQuery(),
},

'actor-reflect':{
'prompt':"""First, identify up to 2 types of information other than observation and action (from the gameplay history) that may be important for the subgoal '$db.subgoals.subgoal$', and output them in a concise list.

Secondly, summarize the gameplay history using a table.
Use 'TBD' for the action corresponding to the current step, and include the observations, action (including number of steps and result), and the identified information in the table.

Finally, analyze and evaluate each action in '$db.allowed_actions$'. Utilize spatial and temporal reasoning with the table to determine if the action is safe, if it leads to an unexplored state.
Format the analysis as a Json dictionary of the following format, and write "NA" for not applicable fields:
```
{
"$action": { # The action name
"spatial_relation_with_current_obs": $answer, # How does this action spatially relate to objects in the observation? Focus on important objects (1 sentence)
"alignment_with_history":{
"temporal": $answer, # How does this action relate with the most recent historical trajectory temporally? Explain your reasoning by connecting with the history table. (1 sentence)
"spatial": $answer, # How does this action relate to the most recent historical path spatially? Explain your reasoning by connecting with the history table. (1 sentence)
},
"risk": $answer, # list potential risks or hazards associated with the action as concisely as possible
"benefit": $answer, # list potential benefits or advantages associated with the action as concisely as possible
},
...
}
""",
'dep':['top-subgoal', 'subgoal_analysis', 'obs_current_actions', 'obs_obj', 'obs_inv', 'obs_vit'],
'after': ['unknown',],
'shorthand': "Trajectory summary and action analysis",
'compose': ComposeActorEfficiencyPrompt(),
'query': query_spatial,
},

'actor-plan-sketch':{
'prompt':"""First, describe the current observation/surroundings in with a focus on things related to the subgoal '$db.subgoals.subgoal$' and answer the following questions (no more than 1 sentence per-answer):
Out of the goal-relevant objects, which ones are easily reachable and which ones are harder to reach?
Are there direct obstacles in the way?
Are there risks in the way? Are they addressable?

Then, determine if the previous plan '$db.action_summary.plan-sketch$' still applies for the subgoal based on the criteria and the expiration condition.
Relevance criteria: $db.action_summary.relevance-crieria$
Expiration condition: $db.action_summary.expiration-condition$

If the plan still applies, examine the current observation for any obstacles, hazards, or unexpected scenarios and reason spatially how they may be addressed.
If necessary, update only the 'details' of the previous plan to achieve the target '$db.action_summary.target$' of the plan.

If the plan does not apply, explain your reasoning, and write a new plan-sketch.
Reason spatially and temporally with the current observation, the gameplay history, and the action analysis.

Finally, output the updated plan-sketch. The plan-sketch details should concretely describe a procedure or a specific direction to follow, and must be a Json dictionary of the following format:
```
{
"plan-sketch": $plan-sketch, # A 1-line summary of the plan-sketch
"details", # Concrete description of what procedure or a specific direction to follow. Do not offer multiple options or possibilities.
"target", # Concisely write the target of the plan
"relevance-criteria": $relevance_criteria, # The criteria that the plan is relevant to the current situation
"expiration-condition": $expiration_condition, # The condition that may cause the plan to expire, like specific changes in the observation or the inventory, or after a certain number of steps.
"notes": $notes # Anything about the current situation or reasoning that may be meaningful to remember for the future.
}
```
""",
'dep':['planner_unexpected', 'planner_mistake', 'top-subgoal', 'subgoal_analysis', 'obs_obj', 'obs_inv', 'obs_vit', 's-action', 'obs_current_actions', 'actor-reflect'],
'shorthand': "Plan-sketch",
'compose': ComposeActorPlannerPrompt(),
'query': query_spatial,
'after_query': ActionSummaryAfterQuery(),
},

'actor-actions':{
'prompt':"""Given the target: $db.action_summary.target$
First, describe the current observation/surroundings and identify any hazards.
Examine the observation/surroundings any obstacles that may interfere with achieving the target.

- Plan-sketch: $db.action_summary.plan-sketch$
- Plan details: $db.action_summary.details$

Discuss how to spatially address or evade the hazards and obstacles based on analysis of the observation, target, and the plan-sketch.

Then, reason with the instruction manual and the current observation, and identify a sequence of actions to achieve the target.

The sequence of identified actions should start from current step and only include actions for up to the first 6 steps, without skipping any steps.
Think step by step and explain your reasoning before writing down the sequence of actions.

Finally, group repeated actions together to speed up the game, and explicitly state the number of repeats.

Note: The player's reach is 1 step to the front.
""",
'dep':['obs_inv', 'obs_vit', 's-action', 'actor-reflect', 'obs_current_actions', 'obs_obj'],
'after':['actor-plan-sketch'],
'shorthand': "Actor plan and reasoning",
'compose': ComposeActorReasoningPrompt(),
'query': query_spatial,
},

'actor-final':{
'prompt':"""Examine the actor plan and reasoning and identify the first action (only the exact action name from the list of all actions in the manual) in the plan. 
Then, if the identified action is *explicitly stated as a repeat*, write the number of times this action should be repeated.
Finally, identify if there are any observed/nearby hazards or obstacles, no matter if they interfere with the plan.
Format the output as a Json dictionary of the following format:
```
{
"action": $action, # Write only the exact action name from the list of all actions in the manual.
"repeats": $repeats # The number of times this action should be repeated, write 1 if the action is not stated as repeated.
"obstacles": $hazard # [yes/no] Presence of obstacles in the course of the action.
"hazards": $hazard # [yes/no] Presence of hazards in the observation/surroundings, no matter if they interfere with the plan.
}
```
""",
'dep':['obs_obj', 'actor-actions'],
'compose': ComposeActorBarePrompt(),
'query': query_reason,
'after_query': ActionAfterQuery(),
},

's-plan':{
'prompt':"In one sentence, describe the high-level subgoals and the reasoning behind the subgoals.",
'dep':['reflect', 'planner_unexpected', 'planner_mistake', 'top-subgoal', 'subgoal_analysis', 'actor-reflect'],
'shorthand': "Subgoals",
'compose': ComposeSummaryPrompt(),
'query': query_fast,
},

's-mistakes':{
'prompt':"In one sentence, concisely explain any unexpected situations or mistakes encountered recently. Output \"NA\" and end the answer if there's nothing unexpected and no mistakes.",
'dep':['reflect', 'planner_unexpected', 'planner_mistake', 'top-subgoal', 'subgoal_analysis'],
'shorthand': "Mistakes",
'compose': ComposeSummaryPrompt(),
'query': query_fast,
},

}


gamestep_questions = {
    prompts['s-obs-obs']['prompt']:"Observation and inventory",
    prompts['s-obs-vitals']['prompt']:"Vitals",
    prompts['s-action']['prompt']:"Action",
}
delayed_gamestep_questions = {
    prompts['s-action']['prompt']
}

strategy_questions_desc = {
    prompts['subgoals']['prompt']: "Most recent top 3 subgoals",
    prompts['top-subgoal']['prompt']: "Most recent top subgoal and completion criteria",
    # strategy_questions_list[12]: "Unknown information and details",
}
attention_question = prompts['gate']['prompt']
adaptive_strategy_questions = [
    prompts['gate-plan_sketch']['prompt'],
    prompts['top-subgoal']['prompt'],
]
goal_question = prompts['top-subgoal']['prompt']
adaptive_question = prompts['planner-adaptive']['prompt'] 
reflection_skip_questions = [
    'subgoals', 'top-subgoal', 'subgoal_analysis', 'skill', 'planner-adaptive', 'kb-add', 'unknown', 's-mistakes',
]
reflection_skip_questions = [prompts[k]['prompt'] for k in reflection_skip_questions]

kb_questions_desc = {
    prompts['unknown']['prompt']: "Unknown information and details",
}
kb_question = prompts['kb-add']['prompt']
adaptive_kb_questions = [
]

q_act = prompts['actor-final']['prompt']
adaptive_actor_questions = [prompts['actor-reflect']['prompt'], prompts['actor-plan-sketch']['prompt'], prompts['actor-actions']['prompt']]
adaptive_dependencies = ['obs_obj', 'obs_inv', 'obs_vit', 'obs_chg', 'obs_last_act']
adaptive_dependencies = [prompts[k]['prompt'] for k in adaptive_dependencies]



gameplan_questions = {
    prompts['s-mistakes']['prompt']:"Mistakes and Unexpectancies",
    prompts['s-plan']['prompt']:"Subgoals",
}

feedback_questions = [
    # "Concisely make a list of historical observations or findings that could benefit the player. Only include the most important 5 items and do not include anything you already know.",
"""Concisely list up to a total number of 3 significant mistakes or dangerous situations encountered on the current skill.
If there are less than 3, list all of them. If there are no significant encounters, write 'NA'.

For each encounter, offer a brief one-sentence advice to address it using information from the instruction manual.
You advice should be precise and actionable, but should retain generality to be applicable to similar situations/problems in the future.

Do not include anything unrelated to the current skill.""",
]

feedback_questions = [
    # (feedback_questions[0], gamestep_questions, "Past findings", False),
    (feedback_questions[0], gameplan_questions, "Past Notes", True),
]

gameplay_questions = {
    # "Summarize learnings from the gameplay into a list. Do not include anything you already know.": ["How did the player die at the final step?",],
    "Base your answers on the gameplay, create a list of up to 5 most important aspects the player should pay more attention to next time. For each aspect, write a 1-line specification on how the aspected may be addressed/quantified.": ["What did the player accomplish?", "How did the player die at the final step?", "What did the player fail to accomplish?", "What mistakes did the player make?", ],
}
gameplay_shorthands = {
    # "Summarize learnings from the gameplay into a list. Do not include anything you already know.": "Learnings",
    "Base your answers on the gameplay, create a list of up to 5 most important aspects the player should pay more attention to next time. For each aspect, write a 1-line specification on how the aspected may be addressed/quantified.": "Attention",
}

def build_graph(llm_functions, database={}):

    database['shorthands'] = {}

    # Create graph
    graph = Graph()
    edge_list = []
    order_list = []

    for _, node_info in prompts.items():
        key = node_prompt = node_info['prompt']
        node = SimpleDBNode(key, node_prompt, graph, llm_functions[node_info['query']]['query_model'], node_info['compose'], database, after_query=node_info['after_query'] if 'after_query' in node_info.keys() else None, verbose=True, token_counter=llm_functions[node_info['query']]['token_counter'])
        graph.add_node(node)

        if 'shorthand' in node_info.keys() and node_info['shorthand'] is not None:
            database['shorthands'][key] = node_info['shorthand']

        for dependency in node_info['dep']:
            dependency_name = prompts[dependency]['prompt']
            edge_list.append((dependency_name, key))

        if 'after' in node_info.keys():
            for dependency in node_info['after']:
                dependency_name = prompts[dependency]['prompt']
                order_list.append((dependency_name, key))


    for edge in edge_list:
        graph.add_edge(*edge)
    for order in order_list:
        graph.add_order(*order)

    database['prompts'] = {
        'gamestep_questions': gamestep_questions,
        'delayed_gamestep_questions': delayed_gamestep_questions,
        'strategy_questions_desc': strategy_questions_desc,
        'attention_question': attention_question,
        'adaptive_strategy_questions': adaptive_strategy_questions,
        'goal_question': goal_question,
        'adaptive_question': adaptive_question,
        'kb_questions_desc': kb_questions_desc,
        'kb_question': kb_question,
        'adaptive_kb_questions': adaptive_kb_questions,
        'q_act': q_act,
        'adaptive_actor_questions': adaptive_actor_questions,
        'gameplan_questions': gameplan_questions,
        'feedback_questions': feedback_questions,
        'gameplay_questions': gameplay_questions,
        'gameplay_shorthands': gameplay_shorthands,
        'adaptive_dependencies': adaptive_dependencies,
        'reflection_skip_questions': reflection_skip_questions,
        # 'action_summary': action_summary,
    }
    
    return graph
