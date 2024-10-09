import json
import copy
import re
import agentkit.compose_prompt as cp 


def describe_gamestep(db, qa_history, i, step, step_offset):
    text_desc = ""
    text_desc += "* Step {}:\n".format(i+1+step_offset)
    obs_desc = ""
    delayed_desc = ""
    plan_desc = ""
    for q,h in db['prompts']['gamestep_questions'].items():
        if q in db['prompts']['delayed_gamestep_questions']:
            if i<len(qa_history)-1:
                delayed_desc += "  - {}: {}\n".format(h, qa_history[i+1][q])
            else:
                delayed_desc += "  - {}: {}\n".format(h, "TBD (current step)")
        else:
            obs_desc += "  - {}: {}\n".format(h, step[q])
    for q,h in db['prompts']['gameplan_questions'].items():
        if q in step.keys():
            plan_desc += "  - {}: {}\n".format(h, step[q])
    text_desc += obs_desc + plan_desc + delayed_desc
    return text_desc.strip()

class ComposeBasePrompt(cp.ComposePromptDB):

    def __init__(self, system_prompt=""):
        super().__init__()
        self.shrink_idx = 1

    def compose_gamestep_prompt(self, qa_history, db, step_offset=0, skip_last=False):
        text_desc = ""
        for i,step in enumerate(qa_history):
            if skip_last and i==len(qa_history)-1:
                continue
            text_desc+=describe_gamestep(db, qa_history, i, step, step_offset)
            text_desc += "\n\n"

        if len(text_desc.strip())==0:
            text_desc = "NA"
        return text_desc.strip()

    def before_dependencies(self, messages, db):
        return messages

    def after_dependencies(self, messages, db):
        return messages

    def compose(self, dependencies, prompt):

        msg = [{"role": "system", "content": self.system_prompt}]

        msg = self.before_dependencies(msg, self.node.db)

        msg = self.add_dependencies(msg, dependencies, self.node.db)

        msg = self.after_dependencies(msg, self.node.db)

        prompt, db_retrieval_results = self.render_db(prompt, self.node.db)
        self.node.rendered_prompt = prompt
        self.node.db_retrieval_results = db_retrieval_results

        msg.append({"role": "user", "content": prompt})

        return msg, self.shrink_idx

class ComposeSummaryPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = "Answer questions using only information provided in this session. Pay special attention to the numbers and details. Use concise language and provide accurate and precise answers. No need to answer in full sentences."


class ComposeObservationPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = "Provide accuracte answers by drawing connection between the instruction manual and the player's in-game observation."
    
    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        messages.append({"role": "system", "content": "Player's in-game observation:\n\n{}".format(db['environment']['observation_current'])})
        return messages

class ComposeObservationReasoningPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = "Provide accuracte answers based on the observation, inventory, and vitals."

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        messages.append({"role": "system", "content": "Most recent two steps of the player's in-game observation:\n\n{}".format(db['environment']['observation_2step'])})
        return messages

class ComposeObservationActionReasoningPrompt(ComposeBasePrompt):
        
    def __init__(self):
        super().__init__()
        self.system_prompt = "Provide accuracte answers based on the observation, inventory, and vitals."

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        messages.append({"role": "system", "content": "Player's in-game observation:\n\n{}".format(db['environment']['observation_current'])})
        return messages

class ComposeActionReflectionPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = "Provide accurate answers based on changes in the observation, inventory, and vitals. Pay attention to the unknown information or details. Note that the game does not contain bugs or glitches."

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        return messages

    def after_dependencies(self, messages, db):
        if 'unknowns' in db['kb'] and len(db['kb']['unknowns'])>0:
            kb_desc =  "\n\n".join(["{}:\n\n{}".format(d, db['kb']['unknowns'][q]) for q,d in db['prompts']['kb_questions_desc'].items()])
            messages.append({"role": "system", "content": "{}".format(kb_desc)})
        
        messages.append({"role": "system", "content": "Most recent two steps of the player's in-game observation:\n\n{}".format(db['environment']['observation_2step_with_action'])})
        return messages

class ComposePlannerPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """Assist the player to make the best plan for the game by analyzing current observation, current inventory, current status, the manual, the gameplay history, and game guidance. The game does not contain bugs or glitches, and does not offer additional cues or patterns.
Pay close attention to the details such as the numbers and requirement specifications.
Be vigilant for information or details missing from the manual, and do not make use of or assume anything not in the game, the manual, or the knowledge base.
Finally, make sure that each planned items is both quantifiable and achievable with the game actions, and the requirements are well-addressed."""
        self.shrink_idx = 2

    def before_dependencies(self, messages, db):
        skill_feedback = db['feedback']['skill_feedback'][db['skills']['skill']] if db['skills']['skill'] in db['feedback']['skill_feedback'].keys() else ""
        qa_history = db['history']['qa_history_stream'][-db['history']['qa_history_planner_length']:]
        step_offset = db['environment']['step'] - len(qa_history)

        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})

        if len(db['kb']['knowledge_base'])>0:
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(db['kb']['knowledge_base'], indent=0))})
        
        text_desc = self.compose_gamestep_prompt(qa_history, db, step_offset, skip_last=False)

        messages.append({"role": "system", "content": "Gameplay history for the previous {} steps:\n\n{}".format(len(qa_history), text_desc)})

        if len(db['feedback']['feedback']) > 0:
            messages.append({"role": "system", "content": "General guidance:\n\n{}".format(db['feedback']['feedback'])})
        
        if len(db['history']['qa_history'])>0:
            strategy_desc = "\n\n".join(["## {}\n{}".format(d, db['history']['qa_history'][-1][q]) for q,d in db['prompts']['strategy_questions_desc'].items()])
            kb_desc =  "\n\n".join(["## {}\n{}".format(d, db['kb']['unknowns'][q]) for q,d in db['prompts']['kb_questions_desc'].items()])
        else:
            strategy_desc = "Most recent subgoals: None available. Re-plan necessary."
            kb_desc = ""
        
        messages.append({"role": "system", "content": "{}".format(strategy_desc)})

        if len(skill_feedback) > 0:
            messages.append({"role": "system", "content": "Subgoal guidance:\n\n{}".format(skill_feedback)})
        
        return messages

class ComposePlannerReflectionPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """Analyze the observation, inventory, status, the manual, the gameplay history. The game does not contain bugs or glitches, and does not offer additional cues or patterns.
Be vigilant for information or details missing from the manual."""
        self.shrink_idx = 2
    
    def before_dependencies(self, messages, db):
        qa_history = db['history']['qa_history_stream'][-db['history']['qa_history_planner_reflection_length']:]
        step_offset = db['environment']['step'] - len(qa_history)

        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})

        text_desc = self.compose_gamestep_prompt(qa_history, db, step_offset, skip_last=False)

        messages.append({"role": "system", "content": "Gameplay history for the previous {} steps:\n\n{}".format(len(qa_history), text_desc)})

        if len(db["history"]["qa_history"])>0:
            strategy_desc = "\n\n".join(["## {}\n{}".format(d, db["history"]["qa_history"][-1][q]) for q,d in db['prompts']['strategy_questions_desc'].items()])
            kb_desc =  "\n\n".join(["## {}\n{}".format(d, db['kb']['unknowns'][q]) for q,d in db['prompts']['kb_questions_desc'].items()])
        else:
            strategy_desc = "Most recent subgoals: NA"
            kb_desc = ""
        
        messages.append({"role": "system", "content": "{}".format(strategy_desc)})

        return messages

def print_skill_library(skill_library):
    output = {
        k: v['skill_desc'] for k,v in skill_library.items()
    }
    return json.dumps(output, indent=0)

class ComposeSkillPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """Build a skill library to help the player ace the game. The skills should exmplifies high simplicity, granularity, and reusability. Pay close attetion to actions allowed by the game. All skills must be actionable and executable."""

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        return messages

    def after_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Skill library\n\n```{}```".format(print_skill_library(db['skills']['skill_library']))})
        return messages

class ComposeKBAddPrompt(ComposeBasePrompt):
        
    def __init__(self):
        super().__init__()
        self.system_prompt = """Reason with the instruction manual, the observation, the gameplay history.
Gather accuracte information about the 'unknown information and details'. Pay attention to past failures and make sure to separate assumptions from facts.
Reason carefully and precisely."""

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        return messages
    
    def after_dependencies(self, messages, db):
        if 'unknowns' in db['kb'] and len(db['kb']['unknowns'])>0:
            kb_desc =  "\n\n".join(["{}:\n\n{}".format(d, db['kb']['unknowns'][q]) for q,d in db['prompts']['kb_questions_desc'].items()])
        else:
            kb_desc = "Unknown information and details: NA"
        
        messages.append({"role": "system", "content": "{}".format(kb_desc)})
        return messages

class ComposeRewritePrompt(ComposeBasePrompt):
        
    def __init__(self):
        super().__init__()
        self.system_prompt = """Rewrite information accurately and precisely according to the user instructions. Pay close attention to the numbers, the details, and the format of the output."""

class ComposeKBReasonPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = """Uncover previous unknown information from the player's gameplay and observation. Be vigilant for information or details missing from the manual. Focus on the details such as the numbers and requirement specifications."""
        self.shrink_idx = 2

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})

        if len(db['kb']['knowledge_base'])>0:
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(db['kb']['knowledge_base'], indent=0))})
    
        if 'unknowns' in db['kb'] and len(db['kb']['unknowns'])>0:
            kb_desc =  "\n\n".join(["{}:\n\n{}".format(d, db['kb']['unknowns'][q]) for q,d in db['prompts']['kb_questions_desc'].items()])
        else:
            kb_desc = "Unknown information and details: NA"
        
        messages.append({"role": "system", "content": "{}".format(kb_desc)})
        return messages

class ComposeActorPlannerPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = """Your task is to assist the player to accomplish the current top subgoal for a game. Find the best action by reasoning with the previous plan, the manual, the gameplay history, and guidances."""

    def before_dependencies(self, messages, db):
        skill_feedback = db['feedback']['skill_feedback'][db['skills']['skill']] if db['skills']['skill'] in db['feedback']['skill_feedback'].keys() else ""
        feedback = db['feedback']['feedback']
        knowledge_base = db['kb']['knowledge_base']

        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        if len(feedback) > 0:
            messages.append({"role": "system", "content": "General guidance:\n\n{}".format(feedback)})

        if len(skill_feedback) > 0:
            messages.append({"role": "system", "content": "Past mistakes:\n\n{}".format(skill_feedback)})

        if len(knowledge_base)>0:
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(knowledge_base, indent=0))})

        messages.append({"role": "system", "content": "Most recent two steps of in-game observation\n\n{}".format(db['environment']['observation_2step_with_action'])})
        self.shrink_idx = len(messages)
        return messages
    
    def after_dependencies(self, messages, db):
        if db['skills']['skill'] is not None and db['skills']['skill'] == db['skills']['skill_old']:
            messages.append({"role": "system", "content": "Previous plan:\n\n{}".format(json.dumps(db['action_summary'], indent=0))})
            messages.append({"role": "system", "content": "Notes from the last step:\n\n{}".format(db['action_notes'])})
        else:
            messages.append({"role": "system", "content": "Previous plan: Not available, please re-plan."})
            messages.append({"role": "system", "content": "Notes from the last step: NA"})
            db['action_summary'] = {
                "plan-sketch": "NA",
                "details": "NA",
                "target": "NA",
                "relevance-criteria": "NA",
                "expiration-condition": "NA",
            }
            db["action_notes"] = "NA"
        return messages

class ComposeActorEfficiencyPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = """Your task is to assist the player to accomplish the current top subgoal for a game.
Base your answer on the gameplay history, instruction manual, and knowledge base.
Think spatially and temporally, and keep track relative/absolute positions and timestep.
Do not make use of or assume anything not in the game, the manual, or the knowledge base.
The game does not contain bugs or glitches, and does not offer additional cues or patterns."""
        self.shrink_idx = 2

    def before_dependencies(self, messages, db):
        qa_history = db['history']['qa_history_stream'][-db['history']['qa_history_actor_length']:]
        step_offset = db['environment']['step'] - len(qa_history)

        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})

        text_desc = self.compose_gamestep_prompt(qa_history, db, step_offset, skip_last=False)

        messages.append({"role": "system", "content": "Gameplay history for the previous {} steps:\n\n{}".format(len(qa_history), text_desc)})

        if len(db['kb']['knowledge_base'])>0:
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(db['kb']['knowledge_base'], indent=0))})
        
        return messages

class ComposeActorReasoningPrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = """Your task is to assist the player to accomplish the current top subgoal for a game. Start with the current state, and identify what actions to take by reasoning with the previous plan, the manual, the gameplay history, and guidance. Do not make use of or assume anything not in the game, the manual, or the knowledge base. The game does not contain bugs or glitches, and does not offer additional cues or patterns."""
        self.shrink_idx = 2

    def before_dependencies(self, messages, db):
        knowledge_base = db['kb']['knowledge_base']

        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})

        if len(knowledge_base)>0:
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(knowledge_base, indent=0))})

        messages.append({"role": "system", "content": "Most recent two steps of in-game observation\n\n{}".format(db['environment']['observation_2step_with_action'])})
        return messages

class ComposeActorBarePrompt(ComposeBasePrompt):

    def __init__(self):
        super().__init__()
        self.system_prompt = """Precisely answer the following questions based on the context provided."""

    def before_dependencies(self, messages, db):
        messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(db['environment']['manual'])})
        return messages


def compose_filtered_gamestep_prompt(qa_history, attention_rounds, game_questions, plan_questions=None, db=None, step_offset=0, skip_last=False):
    text_desc = ""
    for i,step in enumerate(qa_history):
        if (skip_last and i==len(qa_history)-1) or (i not in attention_rounds):
            continue
        text_desc+=describe_gamestep(db, qa_history, i, step, step_offset)
        text_desc += "\n\n"

    if len(text_desc.strip())==0:
        text_desc = "NA"
    return text_desc.strip()

def compose_feedback_prompt(CTXT_dict, qa_history, current_objective, questions):

    CTXT = CTXT_dict['CTXT']
    db = CTXT_dict['db']
    attention_rounds = CTXT_dict['attention_rounds']
    step_offset = CTXT_dict['step_offset'] - len(CTXT_dict['qa_history'])

    messages_feedback = [
       {"role": "system", "content" : "Provide concrete feedback for the player using concise language. The feedback must be on the current skill."}
    ]
    
    messages_feedback.append({"role": "system", "content": "Instruction manual:\n\n{}".format(CTXT)})

    messages_feedback.append({"role": "system", "content": ""})
    # messages_feedback.append({"role": "system", "content": "Your gameplay:\n\n{}".format(text_desc)})

    messages_feedback.append({"role": "system", "content": "Current skill:\n{}".format(current_objective)})

    messages_feedback.append({"role": "user", "content": ""})

    messages = {}
    for question, dependencies, shorthand, filtered in questions:
        if filtered:
            text_desc = compose_filtered_gamestep_prompt(qa_history, attention_rounds, dependencies, plan_questions=None, db=db)
        else:
            raise Exception("Not implemented")
            # text_desc = compose_gamestep_prompt(qa_history, db, step_offset)
        msg = copy.deepcopy(messages_feedback)
        msg[2]['content'] = "Gameplay history:\n\n{}".format(text_desc)
        msg[-1]['content'] = question
        messages[shorthand]=msg

    return messages, 2



def compose_gameplay_prompt(CTXT_dict, qa_history, Q_CTXT, question):
    CTXT = CTXT_dict['CTXT']
    db = CTXT_dict['db']
    attention_rounds = CTXT_dict['attention_rounds']

    messages = [
       {"role": "system", "content" : "The player just finished playing the game of crafter. Provide concrete feedback for the player's gameplay using very concise language."}
    ]
    
    messages.append({"role": "system", "content": "You already know:\n\n{}".format(CTXT)})
    
    # text_desc = "\n\n".join(["Step {}:\n{}".format((i+1)*int(granularity), "\n".join(["{}: {}".format(h, step[q]) for q,h in gamestep_questions.items()])) for i,step in enumerate(qa_history)])
    text_desc = compose_filtered_gamestep_prompt(qa_history, attention_rounds, db['prompts']['gamestep_questions'], plan_questions=db['prompts']['gameplan_questions'], db=db)

    messages.append({"role": "system", "content": "Gameplay:\n\n{}".format(text_desc)})

    if len(Q_CTXT)>0:
        for q,a in Q_CTXT:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    return messages, 2