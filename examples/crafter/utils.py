import os,pickle
import json
import traceback

def parse_tuple(answer):
    try:
        start = answer.index("(")
        end = answer.rindex(")")
        result = eval(answer[start:end+1])
        return result, None
    except:
        pass
    try:
        # extract the tuple from the string
        start_index = answer.index("(")
        # print(start_index, answer[start_index:])
        tup = []
        c = 1
        tracking = None
        start = start_index+1
        for i in range(start_index+1, len(answer)):
            if answer[i] in {'(','{'}:
                c+=1
            elif answer[i] in {')','}'}:
                c-=1
            elif answer[i] in {"'",'"'} and tracking == None:
                c+=1
                tracking = answer[i]
            elif answer[i] == tracking:
                c-=1
                tracking = None
            elif c==1 and answer[i] == ',':
                item = answer[start:i].replace("\"", "").replace("'","").strip()
                if item == "False":
                    item = False
                elif item == "True":
                    item = True
                tup.append(item)
                start = i+1
            if c==0:
                item = answer[start:i].replace("\"", "").replace("'","").strip()
                if item == "False":
                    item = False
                elif item == "True":
                    item = True
                tup.append(item)
                end_index = i
                break
        if len(tup)==0:
            return None, "Error: could not evaluate answer as a tuple"
        return tuple(tup), None
    except Exception:
        return None, "Error: {}".format(traceback.format_exc())

def get_ctxt():
    import pickle
    with open("./crafter_initial_QA.pkl", "rb") as f:
        QA_data = pickle.load(f)

    QA_data.keys()

    choosen_idx = {
        "gameplay":[1, 3,],
        "objective":[1,],
        "actions":[1,],
    }


    for k, v in QA_data.items():
        print("=="*10)
        print(k)
        print()
        print("\n".join(["{}{}. {}".format("-> " if i in choosen_idx[k] else "   ", i, x) for i,x in enumerate(v['questions'])]))

    if os.path.exists("cache/ctxt.pkl"):
        with open("cache/ctxt.pkl", 'rb') as f:
            CTXT = pickle.load(f)
    else:
        import itertools
        from llm_api import get_query
        query_model = get_query("gpt-3.5-turbo-1106")
        def get_list(L, idx):
            if L==[]:
                return []
            if type(L[0]) == str:
                return [L[idx]]
            else:
                return list(itertools.chain.from_iterable([get_list(ll, idx) for ll in L]))

        CTXT = ""
        for k, ll in choosen_idx.items():
            for idx in ll:
                ans_list = get_list(QA_data[k]['answers'], idx)
                CTXT+= QA_data[k]["questions"][idx] + "\n"
                prompt = "Question: {}\n".format(QA_data[k]["questions"][idx]) + "\n".join(ans_list) + "\n\nRemove duplicate items. New Answer:\n"
                answer = query_model(prompt, 0)
                CTXT+= answer
                CTXT+= "\n\n"
        CTXT = CTXT.strip()
        with open("cache/ctxt.pkl", 'wb') as f:
            pickle.dump(CTXT, f)
    CTXT = CTXT.replace("DO NOT answer in LaTeX.", "")

    CTXT = CTXT.replace("Move Up: Flat ground above the agent.", "Move North: Flat ground to the north of the agent.")
    CTXT = CTXT.replace("Move Down: Flat ground below the agent.", "Move South: Flat ground to the south of the agent.")
    CTXT = CTXT.replace("Move Left: Flat ground left to the agent.", "Move West: Flat ground to the west of the agent.")
    CTXT = CTXT.replace("Move Right: Flat ground right to the agent.", "Move East: Flat ground to the east of the agent.")
    # CTXT = CTXT.replace("8. Place Table: Wood in inventory.", "8. Place Table: 2 Wood in inventory.")
    # CTXT = CTXT.replace("9. Place Furnace: Stone in inventory.", "9. Place Furnace: 4 Stone in inventory.")
    CTXT += "\n\nHealth restores automatically over time, independent from food and hydration."
    notes = [
        "Diagonal actions are not supported, only use the four cardinal directions.",
        "The game world is infinitely large and procedurally generated from a fixed random seed.",
        "If you are within close proximity to a zombie, it will chase you. You must kill the zombie to survive.",
        "When sleeping, the player will not be able to take any actions until energy is full and will take triple damage from zombies. Therefore, do not sleep when threats are nearby.",
    ]
    CTXT += "\n\nNotes:\n" + '\n'.join([" - " + x for x in notes])

    CTXT = CTXT.replace("In plain text. List all objects I need to interact/avoid to survive in the game. Use \"I would like to X object Y\" in each step. Replace Y by the actual object, X by the actual interaction.", "List of desired interactions:")
    CTXT = CTXT.replace("I would like to ", " - ")

    CTXT = CTXT.replace("Write all information helpful for the game in a numbered list.", "List of helpful information:")
    CTXT = CTXT.replace("Write all game objectives numbered list. For each objective, list its requirements.", "List of game achievements and their requirements:")
    CTXT = CTXT.replace("Write all actions as a numbered list. For each action, list its requirements.", "List of all actions and their requirements:")

    print(CTXT)
    return CTXT

def describe_achievements(info, CTXT):
    new_CTXT = CTXT
    new_CTXT += "\n\nGame Objective: Survive and accomplish as many of the achievements as possible, and always be prepared for threats in the game."
    unaccomplished_list = [k.replace("_", " ") for k,v in info['achievements'].items() if v<1]
    accomplished_list = [k.replace("_", " ") for k,v in info['achievements'].items() if v>0]
    # print(unaccomplished_list)
    new_CTXT += "\nCurrent *accomplished* achievements: " + ", ".join(accomplished_list)
    new_CTXT += "\nCurrent *unaccomplished* achievements: " + ", ".join(unaccomplished_list)
    return new_CTXT