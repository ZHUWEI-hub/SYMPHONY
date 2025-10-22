import itertools
import json
import os
from collections import defaultdict

import numpy as np
from models import gpt, gpt3
import wikienv, wrappers
import requests
import logging
import random

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="train")
env = wrappers.LoggingWrapper(env)

global reflection_map
global failed_trajectories
reflection_map = []
failed_trajectories = []
reflection_gt_answer = "This is the correct answer, please adjust your response based on this answer. The correct answer is:"
flag = False
cannotGnerate = False
gt_answer = ""
isClaude= True

# 种子持久化文件
SEED_FILE = "seed_state.json"

def load_seed():
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, 'r') as f:
            return json.load(f)["current_seed"]
    return 0  # 初始种子

def save_seed(seed):
    with open(SEED_FILE, 'w') as f:
        json.dump({"current_seed": seed}, f)

def get_next_seed():
    current_seed = load_seed()
    current_seed += 1
    save_seed(current_seed)
    return current_seed


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
import math
def compute_E(C: float) -> float:
    if C == 0 or C == 1:
        return 0.0
    return -C * math.log(C) - (1 - C) * math.log(1 - C)

# EMCS EVALUATION
def compute_R(Z: float, C: float) -> float:
    E = compute_E(C)
    return Z * (1 - E)
def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    global reflection_map
    global failed_trajectories
    
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    value_prompt = task.value_prompt_wrap(x, y, unique_trajectories, reflection_map) #evaluate prompt
    logging.info(f"Current: {x}")
    logging.info(f"Current: {y}")
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    logging.info(f"VALUE PROMPT: {value_prompt}")  #

    value_outputs = gpt3(value_prompt, n=n_evaluate_sample, stop=None) # n_evaluate_sample
    logging.info(f"VALUE OUTPUTS: {value_outputs}")
    # value = task.value_outputs_unwrap(value_outputs)
    # logging.info(f"VALUES: {value}")


    score, conf = task.value_outputs_unwrap(value_outputs)

    logging.info(f"{score}，{conf}")
    value = compute_R(score, conf)
    if value < 0.2:
        value = 0.2
    logging.info(f"VALUES: {value}")


    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop ,temperature1):
    global failed_trajectories
    global reflection_map
    global flag
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    if  len(unique_trajectories) > len(reflection_map) and len(unique_trajectories)< 4 :
        print("generating reflections")
        reflection_map = task.generate_self_reflection(unique_trajectories, x)
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, reflection_map)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')

    logging.info(f"PROMPT: {prompt}")
    samples = gpt(prompt, n=n_generate_sample, stop=stop,temperature= temperature1)

    # import re
    # samples_list = []
    #
    # for response in samples:
    #
    #     match = re.search(r'Thought \d+:(.*)', response)
    #     if match:
    #         response = match.group(1)
    #     else:
    #         response = response
    for i in range(len(samples)):
        if samples[i][0][0].startswith("Thought"):
            index = samples[i][0][0].find(":")
            samples[i][0][0] = samples[i][0][0][index + 1:].strip()

    for i in range(len(samples)):
        samples[i][0][0] = y + samples[i][0][0]
    return samples

    # return [y + _ for _ in samples]

def get_unique_trajectories(failed_trajectories, num=5):

    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories

class Node:
    def __init__(self, state, question, parent=None):
        self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.model = ""

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def __str__(self):
        return f"Node(depth={self.depth},model={self.model}, value={self.value:.2f}, visits={self.visits}, reward={self.reward}, em={self.em} ,is_terminal={self.is_terminal} ,thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
            'model': self.model
        }
    
def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    print(f"lines:{lines}")
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue
        
        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")
    
    return '\n'.join(formatted_lines)

def collect_all_nodes(node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(collect_all_nodes(child))
        return nodes


def collect_trajectory(node):
    trajectory = []
    while node:
        trajectory.append(str(node))
        node = node.parent
    return '\n'.join(reversed(trajectory))

def mcts_search(args, task, idx, iterations=30, to_print=True):
    global gpt
    global failed_trajectories
    global reflection_map
    global gt_answer
    global flag
    global reflection_gt_answer
    cannotGnerate = False
    flag = False

    global isClaude

    x = env.reset(idx=idx)
    if to_print:
        print(idx, x)
    root = Node(state=None, question=x) # root
    all_nodes = []
    failed_trajectories = []
    terminal_nodes = []
    reflection_map = []
    # logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')


    s = get_next_seed()
    random.seed(s)


    for i in range(iterations):
        logging.info(f"Iteration {i + 1}...")
        node = select_node(root)

        while node is None or (node.is_terminal and node.reward != 1):
            logging.info(f"Need to backtrack or terminal node with reward 0 found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            if node == -1:
                logging.info("terminal")
                return None, None, None, None, None, None


        if node.is_terminal and node.reward == 1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            return node.state, node.value, all_nodes, node.reward, node.em, node.question

        successnode = expand_node(node, args, task)
        if successnode is not None:
            return successnode.state, successnode.value, [], successnode.reward, successnode.em,successnode.question

        while node.is_terminal or not node.children:
            logging.info(f"Depth limit node found at iteration {i + 1}, reselecting...")
            node = select_node(root)
            if node is None:
                logging.info("no")
                break
            successnode = expand_node(node, args, task)
            if successnode is not None:
                return successnode.state, successnode.value, [], successnode.reward, successnode.em ,successnode.question


        value = evaluate_node(node, args, task)
        # Find the child with the highest value
        reward, terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=4) # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！

        if cannotGnerate:
            return None,None,None,None,None,None
        #rollout
        terminal_nodes.append(terminal_node)

        if terminal_node.reward == 1:
            logging.info("SUCCESSFUL TRAJECTORY FOUND DURING SIMULATION")
            return terminal_node.state, terminal_node.value, [], terminal_node.reward, terminal_node.em, terminal_node.question

        backpropagate(terminal_node, reward)
        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]

        # Check for terminal nodes with a reward of 1
        terminal_nodes_with_reward_1 = [node for node in collect_all_nodes(root) if node.is_terminal and node.reward == 1]
        if terminal_nodes_with_reward_1:
            logging.info(f"Terminal node with reward 1 found at iteration {i + 1}")
            best_node = max(terminal_nodes_with_reward_1, key=lambda x: x.value)
            return best_node.state, best_node.value, all_nodes, best_node.reward, best_node.em, best_node.question
        for j, (node, value) in enumerate(all_nodes):
            logging.info(f"Node {j+1}: {str(node)}")

        logging.info(f"State of all_nodes after iteration {i + 1}: {all_nodes}")

    all_nodes_list = collect_all_nodes(root)
    all_nodes_list.extend(terminal_nodes)
    best_child = max(all_nodes_list, key=lambda x: x.reward)
    failed_trajectories = []
    if best_child.reward == 1:
        logging.info("Successful trajectory found")
    else:
        logging.info("Unsuccessful trajectory found")
    if best_child is None:
        best_child = root
    return best_child.state, best_child.value, all_nodes, best_child.reward, best_child.em, best_child.question

def select_node(node):

    while node and node.children:
        logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
        
        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]
        
        if len(terminal_children) == len(node.children):
            logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
            if node.parent:
                node.parent.children.remove(node)
            node = node.parent
            if node is None:
                logging.info("no terminal")
                return -1
            continue  
        
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1

        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

        while node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
            
        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
        
    return node  # This will return None if all paths from the root are exhausted

def expand_node(node, args, task):
    if node.depth >= 7:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, args.n_generate_sample,args.temperature)
    for state in new_nodes:
            if state.is_terminal and state.reward == 1:
                from run import llm_manager
                llm_manager.update_reward(state.model, state.reward)
                return state

    node.children.extend(new_nodes)

def rollout(node, args, task, idx, max_depth=4):
    from run import llm_manager
    logging.info("ROLLING OUT")
    depth = node.depth
    n = args.n_generate_sample
    global cannotGnerate

    rewards = [0]
    while not node.is_terminal and depth < max_depth:
        # Generate new states
        logging.info(f"ROLLING OUT {depth}")
        new_states = []
        values = []
        temperature1 = args.temperature
        count = 0
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, n, temperature1)
            if len(new_states) == 0:# generate_new_states
                temperature1 = args.temperature - (0.01 * count)
                if temperature1 <= 0:
                    temperature1 = 0.1
                    cannotGnerate = True
                    break
                logging.info(f"当前模型的温度为: {temperature1}")
            count += 1


        terminal_state = None

        if cannotGnerate:
            return 0, None

        for state in new_states:
            if state.is_terminal:
                if state.is_terminal and state.reward == 1:
                    return state.reward, state
                terminal_state = state

        if terminal_state is not None:
            return terminal_state.reward, terminal_state


        index_node = {}
        action_groups = defaultdict(list)
        for i in range(0, len(new_states)):
            index_node[new_states[i]] = i
            action_groups[new_states[i].state['action']].append(new_states[i])

        max_action = 0
        max_node = None
        for action, nodes in action_groups.items():
            if len(nodes) > max_action:
                max_action = len(nodes)
                max_node = nodes

        if max_action >= args.n_generate_sample - 1 and len(new_states) > 2:
            for i in range(0, len(new_states)):
                values.append(0.7)
            for child in max_node:
                values[index_node[child]] = 0.9
                logging.info(f"no evaluation:{new_states},values为{values}")
        else:
            logging.info(f"yes evaluation:{new_states}")

            child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
            #new_state = new_state[0]


            while len(values) == 0:
                values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)

        for index in range(len(values)):
            llm_manager.update_reward(new_states[index].model, values[index])

        max_value_index = values.index(max(values))
        rewards.append(max(values))
        node = new_states[max_value_index]
        depth += 1
        if depth == max_depth:
            rewards = [-1]
    
    logging.info("ROLLOUT FINISHED")
    return sum(rewards) / len(rewards), node

def generate_new_states(node, args, task, n, temperature1):
    global failed_trajectories
    #---------------------------
    global flag
    #---------------------------

    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, f"Thought {node.depth + 1}: ", n, prompt_sample=args.prompt_sample, stop="Observation", temperature1=temperature1)
    logging.info(f"SAMPLED ACTION: {sampled_actions}")
    tried_actions = []
    
    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        new_state = node.state.copy()



        # thought_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith(f"Thought {node.depth + 1}")), '')
        # action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)


        thought_line = next(
            (line.split(":")[1].strip()
             for line in action[0][0].split("\n")
             if line.startswith(f"Thought {node.depth + 1}")),
            ''
        )

        action_line = next(
            (line.split(":")[1].strip()
             for line in reversed(action[0][0].split("\n"))
             if line.startswith("Action") and ":" in line),
            None
        )



        # Use thought and action to form a unique key
        unique_key = f"{thought_line}::{action_line}"
        
        if unique_key in unique_states:
            continue

        tried_actions.append(action_line)
        
        if action_line:
            action_type = action_line.split('[')[0] if '[' in action_line else action_line
            action_param = action_line.split('[')[1].split(']')[0] if '[' in action_line else ""

            obs, r, done, info = step(env, f"{action_type.lower()}[{action_param}]")

            # Update the new state dictionary
            new_state['thought'] = thought_line
            new_state['action'] = action_line
            new_state['observation'] = obs

            new_node = Node(state=new_state, question=node.question, parent=node)
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            new_node.depth = node.depth + 1

            new_node.model = action[1]

            if r == 1:
                new_node.em = info.get('em')
            else:
                new_node.reward = 0

            unique_states[unique_key] = new_node   # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")
            logging.info(f"Feedback: {info}")  #

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
                #print(trajectory)
                #if f"{action_type.lower()}[{action_param}]" not in failed_trajectories.values():
                failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_type.lower()}[{action_param}]"})

            if new_node.is_terminal and r == 1:
                break

    return list(unique_states.values())  # Return unique nodes as a list，node


def evaluate_node(node, args, task):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]


    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)  # ！！！！
    
    logging.info(f"Length of votes: {len(votes)}")
    logging.info(f"Length of node.children: {len(node.children)}")
    
    # Pre-allocate votes list
    votes = votes + [0] * (len(node.children) - len(votes))
    for i, child in enumerate(node.children):
        child.value = votes[i]
        from run import llm_manager
        llm_manager.update_reward(child.model, child.value)
    # max_vote = max(votes) if votes else 1
    # if max_vote == 0:
    #     max_vote = 1  # Avoid division by zero
    
    # terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
    # for i, condition in enumerate(terminal_conditions):
    #     if condition == 1:
    #         votes[i] = max_vote + 1
    
    # for i, child in enumerate(node.children):
    #     child.value = votes[i] / max_vote  # Now safe from division by zero
    
    return sum(votes) / len(votes) if votes else 0


def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)

def backpropagate(node, value):
    while node:
        node.visits += 1
        if node.is_terminal:
            if node.reward == 0:
                node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                logging.info(f"Backpropagating with reward 0 at depth {node.depth}. New value: {node.value}.")
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits
                logging.info(f"Backpropagating with reward 1 at depth {node.depth}. New value: {node.value}.")
        else:
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

        node = node.parent

def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['thought']:
            new_segment.append(f"Thought {node.depth}: {node.state['thought']}")
        if node.state['action']:
            new_segment.append(f"Action {node.depth}: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
            new_segment.append(f"Observation {node.depth}: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n'.join(reversed(trajectory))