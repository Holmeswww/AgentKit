import argparse, os

import crafter
import tqdm
import numpy as np
import pandas as pd
import copy
import json
import datetime
import pickle
from colorama import Fore, Back, Style
from colorama import init
import agentkit
import agentkit.utils as utils
init(autoreset=True)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--outdir', type=str, default='logs/test_env', help='output directory')
parser.add_argument('--Eps', type=int, default=1, help='Total number of episodes to run')
parser.add_argument('--temperature', type=float, default=0., help='LLM Temperature')
parser.add_argument('--wandb_log_interval', type=int, default=25, help='some integer')
parser.add_argument('--granularities', type=int, nargs='+', default=[500], help='list of integers')
parser.add_argument('--feedback_granularity', type=int, default=3, help='some integer')
parser.add_argument('--planner_reflection_granularity', type=int, default=25, help='some integer')
parser.add_argument('--actor_reflection_granularity', type=int, default=5, help='some integer')
parser.add_argument('--kb_refine_granularity', type=int, default=50, help='some integer')
parser.add_argument('--llm_plan_accurate', type=str, default="gpt-4-turbo-2024-04-09", help='LLM')
parser.add_argument('--llm_plan', type=str, default="gpt-4-turbo-2024-04-09", help='LLM')
parser.add_argument('--llm_spatial', type=str, default="gpt-4-0613", help='LLM')
parser.add_argument('--llm_fast', type=str, default="gpt-3.5-turbo", help='LLM')
parser.add_argument('--llm_fast_accurate', type=str, default="gpt-3.5-turbo-0125", help='LLM')
parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode')
parser.add_argument('--resume_id', type=str, default=None, help='resume wandb run id')

args = parser.parse_args()
args.verbose = not args.quiet

# make the saves folder if it doesn't exist
os.makedirs('saves', exist_ok=True)
os.makedirs('prints', exist_ok=True)

if args.resume_id is not None:
    with open("saves/{}.pkl".format(args.resume_id), 'rb') as f:
        args = pickle.load(f)['args']

from crafter_description import describe_frame, action_list

env = crafter.Env(area=(256, 256))
action_space = env.action_space


from utils import get_ctxt, describe_achievements
MANUAL = get_ctxt()

from build_graph_new import build_graph
from compose_prompt import compose_feedback_prompt, compose_gameplay_prompt
from agentkit.llm_api import get_query, get_token_counts 
from functools import partial

import wandb
from wandb.sdk.data_types.trace_tree import Trace
if args.resume_id is not None:
    wandb.init(id=args.resume_id, resume="allow", project='AgentKit', config=args.__dict__)
else:
    wandb.init(project='AgentKit', config=args.__dict__)
    wandb.run.log_code(".") # save the code in the wandb run

if args.verbose:
    def qprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    def qprint(*args, **kwargs):
        pass

achievements = [
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up',
]

if args.resume_id is not None:
    with open("saves/{}.pkl".format(args.resume_id), 'rb') as f:
        pkl_save = pickle.load(f)
    eps = pkl_save['eps']
    database = pkl_save['database']
else:
    eps = 0
    database = {}
    database['kb'] = {
        'unknowns': {},
        'knowledge_base': {},
    }
    database['subgoals'] = {
        'subgoal': "NA",
        'guide': "NA",
    }
    database['reflection'] = {
        "unexpected": [],
        "mistake": [],
        "correction": [],
        "confusion": [],
        "all": [],
    }
    database['history'] = {}
    database['skills'] = {
        "skill_library": {},
        "skill": None,
        "skill_old": None,
    }
    database['feedback'] = {
        'skill_feedback': {},
        'feedback': "",
    }

query_fast = get_query(args.llm_fast)
query_fast_accurate = get_query(args.llm_fast_accurate)
query_reason = get_query(args.llm_plan)
query_spatial = get_query(args.llm_spatial)
query_plan_accurate = get_query(args.llm_plan_accurate)

llm_functions = {
    'query_fast': {'query_model':query_fast, 'token_counter':query_fast.count_tokens},
    'query_fast_accurate': {'query_model':query_fast_accurate, 'token_counter':query_fast_accurate.count_tokens},
    'query_reason': {'query_model':partial(query_reason, max_gen=2048, temp=args.temperature), 'token_counter':query_reason.count_tokens},
    'query_plan_accurate': {'query_model':partial(query_plan_accurate, max_gen=2048, temp=args.temperature), 'token_counter':query_plan_accurate.count_tokens},
    'query_spatial': {'query_model':partial(query_spatial, max_gen=1500, temp=args.temperature), 'token_counter':query_spatial.count_tokens},
}
graph = build_graph(llm_functions, database)

table_questions = list(graph.nodes.keys())

columns=["Step", "OBS", "Reward", "Return"] + ["Action", "Repeats", "Skills", "Knowledge Base", "Env step"] + table_questions

while(eps<args.Eps):

    graph = build_graph(llm_functions, database)
    if args.resume_id is not None:
        done = pkl_save['done']
        step = pkl_save['step']
        env_step = pkl_save['env_step']
        env = pkl_save['env']
        trajectories = pkl_save['trajectories']
        qa_history = pkl_save['qa_history']
        gameplay_history = pkl_save['gameplay_history']
        R = pkl_save['R']
        OBS = pkl_save['OBS']
        a = pkl_save['a']
        obs = pkl_save['obs']
        reward = pkl_save['reward']
        info = pkl_save['info']
        last_log = pkl_save['last_log']

        rollout_history = pkl_save['rollout_history']
        skill_length = pkl_save['skill_length']
        skill_history = pkl_save['skill_history']
        

        achievement_table = pkl_save['achievement_table']
        feedback_table = pkl_save['feedback_table']
        root_span = pkl_save['root_span']
        args.resume_id = None
        adaptive_answers = pkl_save['adaptive_answers']
        past_actions = pkl_save['past_actions']

    else:
        done = False
        env_step = step = 0
        env.reset()
        trajectories = []
        qa_history = []
        skill_history = {}
        database['history']= {}
        database['action_repeats']=0
        database['reflection'] = {
            "unexpected": [],
            "mistake": [],
            "correction": [],
            "confusion": [],
            "all": [],
        }
        
        gameplay_history = {str(g):[] for g in args.granularities}
        R = 0
        OBS = []
        a = action_list.index("noop")
        obs, reward, done, info = env.step(a)
        R += reward
        OBS.append(obs.copy())

        achievement_table = wandb.Table(columns=achievements)
        feedback_table = wandb.Table(columns=["Skill", "feedback"] + ["attn"])
        root_span = Trace(
            name="Agent_eps{}".format(eps),
            kind="agent",
            start_time_ms=round(datetime.datetime.now().timestamp() * 1000),
            metadata={"eps": eps,}
        )
        # Add wandb root span
        graph.set_wandb_root_span(root_span)

        last_log = 0

        rollout_history = []
        database['skills']['skill'] = database['skills']['skill_old'] = None
        skill_length = 1
        adaptive_answers = "NA"
        past_actions = []

    while not done:
        
        last_act_desc, desc, _ = describe_frame(info, database['action_repeats'])
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Action " + "--"*10 + Style.RESET_ALL)
        qprint(last_act_desc)
        qprint()
        qprint(Fore.BLACK + Back.WHITE + Style.BRIGHT + "=="*15 + " Step: {}, Env step: {}, Reward: {} ".format(step, env_step, R) + "=="*15 + Style.RESET_ALL)
        # qprint(desc)
        new_row = [step, desc, reward, R,]
        wandb.log({"eps-{}-metric/reward".format(eps): reward, "eps-{}-metric/return".format(eps): R, 'eps-{}-metric/step'.format(eps): step, 'eps-{}-metric/env_step'.format(eps): env_step}, commit=False)
        tokens_table = wandb.Table(columns=["node", "calls", "prompt", "completion"])
        for node in graph.nodes.values():
            if node.token_counter is not None:
                counts = node.get_token_counts()
                tokens_table.add_data(*[node.key, counts['calls'], counts['prompt'], counts['completion']])
        wandb.log({"token_counts": tokens_table,}, commit=False)
        wandb.log(get_token_counts())

        if len(trajectories)>0:
            trajectories[-1][1] = last_act_desc
        trajectories.append([step, None, desc])
        text_obs_no_act = "\n\n".join(["== Gamestep {}{} ==\n\n".format(i, "" if i!=trajectories[-1][0] else " (current)",) + "{}".format(d) for i, _, d in trajectories[-2:]])
        text_obs = "\n\n".join(["== Gamestep {}{} ==\n\n".format(i, "" if i!=trajectories[-1][0] else " (current)",) + "{}{}".format(d, "\n\nAction:\n{}".format(a) if a is not None else "") for i, a, d in trajectories[-2:]])
        qprint(text_obs)

        database['environment'] = {
            'manual': describe_achievements(info, MANUAL),
            'observation_2step': text_obs_no_act,
            'observation_2step_with_action': text_obs,
            'observation_current': desc,
            'step': step,
        }
        qa_history_stream = copy.copy(qa_history)
        qa_history_stream.append(graph.get_streaming_history())
        database['history'] = {
            'qa_history': qa_history,
            'qa_history_stream': qa_history_stream,
            'qa_history_actor_length': min(args.actor_reflection_granularity, max(3, skill_length)),
            'qa_history_planner_length': args.planner_reflection_granularity,
            'qa_history_planner_reflection_length': 3,
        }

        # Printing
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Actor Plans " + "--"*10 + Style.RESET_ALL)
        if 'action_summary' in database.keys():
            qprint(Style.DIM+json.dumps(database['action_summary'], indent=2) + Style.RESET_ALL)
        qprint("Skill: {} -> {}".format(database['skills']['skill_old'], database['skills']['skill']))
        qprint("Past actions:", " -> ".join(past_actions))
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Subgoal " + "--"*10 + Style.RESET_ALL)
        if len(qa_history)>0:
            strategy_desc = "\n\n".join(["## {}\n{}".format(d, qa_history[-1][q]) for q,d in database['prompts']['strategy_questions_desc'].items()])
            qprint(Style.DIM+ strategy_desc + Style.RESET_ALL)
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Knowledge Base " + "--"*10 + Style.RESET_ALL)
        qprint(Style.DIM+json.dumps(database['kb']['knowledge_base'], indent=2) + Style.RESET_ALL)
        if 'unknowns_json' in database['kb'].keys():
            qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Unknowns " + "--"*10 + Style.RESET_ALL)
            qprint(Style.DIM+json.dumps(list(database['kb']['unknowns_json'].values())[0], indent=2) + Style.RESET_ALL)
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Skills " + "--"*10 + Style.RESET_ALL)
        qprint(Style.DIM+json.dumps({k: v['skill_desc'] for k,v in database['skills']['skill_library'].items()}, indent=2) + Style.RESET_ALL)
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Skill Feedback " + "--"*10 + Style.RESET_ALL)
        qprint(Style.DIM + json.dumps(database['feedback']['skill_feedback'], indent=2) + Style.RESET_ALL)
        # qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Achievements " + "--"*10 + Style.RESET_ALL)
        # qprint(Style.DIM + describe_achievements(info, MANUAL) + Style.RESET_ALL)

        # Reasoning
        qprint("\n" + Fore.BLACK + Back.GREEN + "--"*10 + " Reasoning " + "--"*10 + Style.RESET_ALL)
        database['skills']['skill_old'] = database['skills']['skill']
        qa_results = graph.evaluate()
        qa_history.append(qa_results)
        
        skill = database['skills']['skill']
        skill_old = database['skills']['skill_old']
        attention_rounds = database['reflection']

        if skill is not None:
            if skill not in skill_history.keys():
                skill_history[skill] = []
            skill_history[skill].append(step)
            if (len(set(skill_history[skill]) & set(attention_rounds['mistake']).union(set(attention_rounds["confusion"])))+1) % args.feedback_granularity == 0:
                CTXT_dict = {
                    "CTXT": describe_achievements(info, MANUAL),
                    "attention_rounds": list(set(skill_history[skill]) & set(attention_rounds["mistake"]).union(set(attention_rounds["confusion"]))),
                    'step_offset': step,
                    'qa_history': qa_history,
                    'db': database,
                }
                messages, shrink_idx = compose_feedback_prompt(CTXT_dict, qa_history, "{}:{}".format(skill, database['skills']['skill_library'][skill]['skill_guide']), database['prompts']['feedback_questions'])
                database['feedback']['skill_feedback'][skill] = ""
                for shorthand, msg in messages.items():
                    answer, _ = llm_functions['query_reason']['query_model'](msg, shrink_idx)
                    # database['feedback']['skill_feedback'][skill] += "{}:\n{}\n\n".format(shorthand, answer)
                    database['feedback']['skill_feedback'][skill] = "{}".format(answer)
                database['feedback']['skill_feedback'][skill] = database['feedback']['skill_feedback'][skill].strip()
                qprint(Fore.MAGENTA + "Feedback for {}:".format(skill) + Style.RESET_ALL)
                qprint(Style.DIM+database['feedback']['skill_feedback'][skill] + Style.RESET_ALL)
                qprint()

        reward = 0
        if info['sleeping']:
            qprint(Fore.RED + "Player is sleeping. We manually take noop until the player's awake to save LLM calls:" + Style.RESET_ALL)
            a = action_list.index("noop")
            database['action'] = a
            rep = 0
            while info['sleeping']:
                obs, rr, done, info = env.step(a)
                qprint(Style.DIM + "====Sleeping: {} Reward: {}====".format(rep+1, rr) + Style.RESET_ALL)
                qprint(Style.DIM + describe_frame(info, 1)[1])
                reward += rr
                env_step += 1
                rep += 1
            database['action_repeats'] = rep
        else:
            a = database['action']
            for _ in range(database['action_repeats']):
                obs, rr, done, info = env.step(a)
                reward += rr
                env_step += 1

                # if the player is blocked, we stop repeating the action
                if 'move' in action_list[a] and not describe_frame(info, 1)[-1]:
                    break

        new_row.append(action_list[database['action']])
        new_row.append(database['action_repeats'])
        new_row.append(json.dumps(database['skills']['skill_library'], indent=2))
        new_row.append(json.dumps(database['kb']['knowledge_base'], indent=2))
        new_row.append(env_step)

        for q in table_questions:
            new_row.append(qa_results[q])

        R += reward
        OBS.append(obs.copy())

        step += 1
        if skill_old != skill:
            past_actions = []
            skill_length = 1
        else:
            past_actions.append(action_list[a])
            skill_length += 1
        achievement_table.add_data(*[info['achievements'][k] for k in achievements])

        rollout_history.append(new_row)


        # Knowledge Base Refinement
        if step % args.kb_refine_granularity == 0 and len(database['kb']['knowledge_base']) > 0:
            messages = [
                {"role": "system", "content" : "Improve the knowledge base. Note that items in the knowledge base should augment the instruction manual, not duplicate it or contradict it. In addition, the knowledge base should not contain duplicate items."}
            ]
            messages.append({"role": "system", "content": "Instruction manual:\n\n{}".format(MANUAL)})
            messages.append({"role": "system", "content": "Knowledge base:\n\n{}".format(json.dumps(database['kb']['knowledge_base'], indent=0))})

            messages.append({"role": "user", "content": """
For each item in the knowledge base, provide a 1-sentence summary of the related manual information if applicable, and determine whether the item should be included or removed from the knowledge base.
Format the output as a JSON dictionary in the following format:
```
{
"item_key": {
    "item_value": $ANSWER,
    "duplicate": $ANSWER, # Is this item a duplicate? [yes/no]
    "manual_summary": $ANSWER, # 1-sentence summary of related manual information. Write "NA" if there's no related manual information.
    "addition": $ANSWER, # Does this item offer additional information to the manual? [yes/no]
    "contradiction": $ANSWER, # Does this item directly contradict the manual_summary? [yes/no]
    }
}
```
""".strip()})
            for _ in range(10):
                result, _ = llm_functions['query_reason']['query_model'](messages, 1)
                parsed_answer, error_msg = utils.extract_json_objects(result)
                if parsed_answer is None or type(parsed_answer[-1]) != dict:
                    messages.append({"role": "assistant", "content": result})
                    messages.append({"role": "user", "content": "Invalid Type: Expecting the last Json object to be dictionary"})
                    continue
                problem = False
                for k, v in parsed_answer[-1].items():
                    if len(v) != 5 or type(v) != dict:
                        messages.append({"role": "assistant", "content": result})
                        messages.append({"role": "user", "content": "Invalid Type: Expecting each value to be a dictionary with 5 keys"})
                        problem = True
                        break
                if problem:
                    continue
                qprint(Fore.MAGENTA + "Refining Knowledge Base:" + Style.RESET_ALL)
                qprint(Style.DIM+json.dumps(parsed_answer[-1], indent=2) + Style.RESET_ALL)
                for k, v in parsed_answer[-1].items():
                    if "no" in [v['addition'].strip(), ] or 'yes' in [v['duplicate'].strip(), v['contradiction'].strip()]:
                        del database['kb']['knowledge_base'][k]
                break



        qprint()
        if step % args.wandb_log_interval == 0 or done:
            if root_span is not None:
                root_span._span.end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
                root_span.log(name="eps-{}-trace".format(eps))
                root_span = None
                graph.set_wandb_root_span(root_span)
            for skill in skill_history.keys():
                feedback_table.add_data(*([skill, "NA", len(set(skill_history[skill]) & set(attention_rounds["mistake"]).union(set(attention_rounds["confusion"])))] if skill not in database['feedback']['skill_feedback'].keys() else [skill, database['feedback']['skill_feedback'][skill], len(set(skill_history[skill]) & set(attention_rounds["mistake"]))]))
            rollouts = wandb.Table(columns=columns, data=copy.deepcopy(rollout_history))
            wandb.log({"eps-{}-rollout/rollout {}~{}".format(eps, last_log, step-1): rollouts, 
                       "eps-{}-achievements/achievements {}~{}".format(eps, last_log, step-1): achievement_table, 
                       "eps-{}-feedback/feedback {}~{}".format(eps, last_log, step-1): feedback_table,
                       "eps-{}-feedback/feedback-current".format(eps): feedback_table,
                       "eps-{}-current/rollout-current".format(eps): rollouts, 
                       "eps-{}-current/achievements-current".format(eps): achievement_table, 
                      })
            achievement_table = wandb.Table(columns=achievements)
            feedback_table = wandb.Table(columns=["Skill", "feedback", "attention_rounds"])
            last_log = step

        with open("saves/{}.pkl".format(wandb.run.id), 'wb') as f:
            pickle.dump({
                'eps': eps,
                'done': done,
                'step': step,
                'env_step': env_step,
                'env': env,
                'trajectories': trajectories,
                'qa_history': qa_history,
                'gameplay_history': gameplay_history,
                'R': R,
                'OBS': OBS,
                'a': a,
                'obs': obs,
                'reward': reward,
                'info': info,
                'last_log': last_log,
                'rollout_history': rollout_history,
                'skill_length': skill_length,
                'skill_history': skill_history,
                'achievement_table': achievement_table,
                'feedback_table': feedback_table,
                'adaptive_answers': adaptive_answers,
                'database': database,
                'past_actions': past_actions,
            }, f)

        if done:
            with open("saves/{}_eps{}.pkl".format(wandb.run.id, eps), 'wb') as f:
                pickle.dump({
                    'eps': eps,
                    'done': done,
                    'step': step,
                    'env_step': env_step,
                    'env': env,
                    'trajectories': trajectories,
                    'qa_history': qa_history,
                    'gameplay_history': gameplay_history,
                    'R': R,
                    'OBS': OBS,
                    'a': a,
                    'obs': obs,
                    'reward': reward,
                    'info': info,
                    'last_log': last_log,
                    'rollout_history': rollout_history,
                    'skill_length': skill_length,
                    'skill_history': skill_history,
                    'achievement_table': achievement_table,
                    'feedback_table': feedback_table,
                    'adaptive_answers': adaptive_answers,
                    'database': database,
                    'past_actions': past_actions,
                }, f)
            break
    
    wandb.log({"eps-{}-achievements/achievements {}~{}".format(eps, last_log, step-1): achievement_table, 
               "eps-{}-feedback/feedback {}~{}".format(eps, last_log, step-1): feedback_table,
               "eps-{}-ALL/rollout-ALL".format(eps): wandb.Table(columns=columns, data=rollout_history),
               "eps-{}-current/achievements-current".format(eps): achievement_table, 
               "eps-{}-feedback/feedback-current".format(eps): feedback_table,
              })
    achievement_table = wandb.Table(columns=achievements)
    last_log = step
    
    # Developer: This part does not seem to help the agent learn better. It's commented out for now.
    # I wrote this part to collect feedback at the end of each round of game, but it seems to be unnecessary.
    # 
    #
    # CTXT_dict = {
    #     "CTXT": MANUAL,
    #     "db": database,
    #     "attention_rounds": database['reflection'],
    # }
    # qa_results = topological_traverse(CTXT_dict, qa_history, database['prompts']['gameplay_questions'], compose_gameplay_prompt, max_gen=1024)
    # end_of_round_table = wandb.Table(columns=[s for s in database['prompts']['gameplay_shorthands'].values()])
    # row = []
    # database['feedback']['feedback'] = ""
    # for q,s in database['prompts']['gameplay_shorthands'].items():
    #     database['feedback']['feedback'] += "{}:\n{}\n\n".format(s, qa_results[q].strip())
    #     row.append(qa_results[q].strip())
    # end_of_round_table.add_data(*row)
    # qprint(Fore.YELLOW + "End of Round FEEDBACK:" + Style.RESET_ALL)
    # qprint(Fore.YELLOW + Style.DIM + database['feedback']['feedback'] + Style.RESET_ALL)
    # wandb.log({"eps-{}-end-feedback/feedback-end-of-round".format(eps): end_of_round_table,})


    eps+=1



wandb.finish()