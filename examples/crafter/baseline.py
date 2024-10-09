import os
os.environ["MINEDOJO_HEADLESS"]="1"
import argparse
import numpy as np
from tqdm import tqdm
import gym
import crafter
from crafter_description import describe_frame, action_list, match_act
from functools import partial
from utils import get_ctxt, describe_achievements
MANUAL = get_ctxt()

parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', type=str, default='yintat-all-gpt-4', help='Name of the LLM')

args = parser.parse_args()

LLM_name = args.llm_name

env = crafter.Env(area=(256, 256))
action_space = env.action_space

# Replace with your own LLM API.
# Note: query_model takes two arguments: 1) message in openai chat completion form (list of dictionaries), 
#                                        2) an index to indicate where the message should be truncated if the length exceeds LLM context length.
from llm_api import get_query
query_model = partial(get_query(LLM_name), max_gen=2048)

def compose_ingame_prompt(info, question, past_qa=[]):
    messages = [
        {"role": "system", "content" : "You’re a player trying to play the game."}
    ]
    
    if len(info['manual'])>0:
        messages.append({"role": "system", "content": info['manual']})

    messages.append({"role": "system", "content": "{}".format(info['obs'])})

    if len(past_qa)>0:
        for q,a in past_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    return messages, 1 # This is the index of the history, we will truncate the history if it is too long for LLM

questions=[
        "What is the best action to take? Let's think step by step, ",
        "Choose the best executable action from the list of all actions. Write the exact chosen action."
    ]

def run():
    env = crafter.Env(area=(256, 256))
    env_steps = 1000000
    num_iter = 2

    rewards = []
    progresses = []
    for eps in tqdm(range(num_iter), desc="Evaluating LLM {}".format(LLM_name)):
        import wandb
        wandb.init(project="Crafter_baseline", config={"LLM": LLM_name, "eps": eps, "num_iter": num_iter, "env_steps": env_steps})
        step = 0
        trajectories = []
        qa_history = []
        progress = [0]
        reward = 0
        rewards = []
        done=False

        columns=["Context", "Step", "OBS", "Score", "Reward", "Total Reward"] + questions + ["Action"]
        wandb_table = wandb.Table(columns=columns)

        env.reset()
        a = action_list.index("noop")
        obs, reward, done, info = env.step(a)
        
        while step < env_steps:
            last_act_desc, desc = describe_frame(info, 1)
            if len(trajectories)>0:
                trajectories[-1][1] = last_act_desc
            trajectories.append([step, None, desc])
            text_obs = "\n\n".join(["== Gamestep {}{} ==\n\n".format(i, "" if i!=trajectories[-1][0] else " (current)",) + "{}{}".format(d, "\n\nAction:\n{}".format(a) if a is not None else "") for i, a, d in trajectories[-2:]])
            info['obs'] = text_obs
            info['manual'] = describe_achievements(info, MANUAL)
            info['reward'] = reward
            info['score'] = sum(rewards)
            new_row = [info['manual'], step, info['obs'], info['score'], reward, sum(rewards)]
            wandb.log({"metric/total_reward".format(eps): sum(rewards), 
                       "metric/score".format(eps): info['score'],
                       "metric/reward".format(eps): reward,
                       })
            
            if done:
                break
            
            qa_history = []
            for question in questions:
                prompt = compose_ingame_prompt(info, question, qa_history)
                answer, _ = query_model(*prompt)
                qa_history.append((question, answer))
                new_row.append(answer)
                answer_act = answer

            a, _, _ = match_act(answer_act)
            if a is None:
                a = action_list.index("noop")
            new_row.append(action_list[a])
            obs, reward, done, info = env.step(a)
            rewards.append(reward)

            step += 1
            wandb_table.add_data(*new_row)

        progresses.append(np.max(progress))
        wandb.log({"rollout/rollout-{}".format(eps): wandb_table, 
                "final/total_reward":sum(rewards),
                "final/episodic_step":step,
                "final/eps":eps,
                })
        del wandb_table
        wandb.finish()

run()