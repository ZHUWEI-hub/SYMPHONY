import json
import logging
import os
import math
import sys
import re
from typing import List, Dict, Any, Tuple

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

sys.set_int_max_str_digits(100000)

react_prompt_header = "Here are some previous solutions and the corresponding test results.\n"
react_prompt_starter = "\n\nYour solution:\n"

work_id = 0


class Node:
    """Node in the MCTS search tree."""
    
    def __init__(self, solution: str, parent=None, context="", depth=0):
        """Initialize a node.
        
        Args:
            solution: Code solution stored in this node
            parent: Parent node
            context: Accumulated context from previous trials
            depth: Depth in the tree
        """
        self.solution = solution
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.context = context
        self.depth = depth
        self.reflection = ""
        self.test_feedback = ""
        self.model = ""

    def uct(self, exploration_weight=1.0):
        """Calculate UCT (Upper Confidence Bound for Trees) value.
        
        Args:
            exploration_weight: Exploration parameter
            
        Returns:
            UCT value
        """
        if self.visits == 0:
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self):
        """Select the best child based on UCT values.
        
        Returns:
            Child node with highest UCT value, or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        """Select the best child based on value only.
        
        Returns:
            Child node with highest value, or None if no children
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.value)

    def update(self, reward: float):
        """Update visit count and value.
        
        Args:
            reward: Reward value to add
        """
        self.visits += 1
        self.value += reward
    

def prune_context_blocks(context: str, max_length: int) -> str:
    """Prune the context to fit within max_length by removing earliest trial blocks.
    
    Args:
        context: Full context string
        max_length: Maximum allowed length
        
    Returns:
        Pruned context string
    """
    if len(context) <= max_length:
        return context
    
    blocks = context.split('Previous Trial')
    
    while len('trial'.join(blocks)) > max_length and blocks:
        blocks.pop(0)
    
    return 'trial'.join(blocks)


def compute_E(C: float) -> float:
    """Compute entropy E(C) for EMCS evaluation.
    
    E(C) = -C * log(C) - (1 - C) * log(1 - C)
    
    Args:
        C: Correctness ratio (0 to 1)
        
    Returns:
        Entropy value
    """
    if C == 0 or C == 1:
        return 0.0
    return -C * math.log(C) - (1 - C) * math.log(1 - C)


def compute_R(Z: float, C: float) -> float:
    """Compute reward R using EMCS evaluation.
    
    R(Z, C) = Z * (1 - E(C))
    
    Args:
        Z: Base reward value
        C: Correctness ratio (0 to 1)
        
    Returns:
        Final reward value
    """
    E = compute_E(C)
    return Z * (1 - E)

def gather_context_from_tree(node: Node) -> Tuple[List[str], List[str]]:
    """Gather feedback and reflections from node to root.
    
    Walk up the tree and collect test feedback and self-reflections from
    each parent node until reaching the root.
    
    Args:
        node: The node to start gathering context from
        
    Returns:
        Two lists containing accumulated feedback and reflections,
        ordered from root to the given node
    """
    accumulated_feedback = []
    accumulated_reflection = []

    while node:
        if node.test_feedback:
            accumulated_feedback.append(node.test_feedback)
        if node.reflection:
            accumulated_reflection.append(node.reflection)
        node = node.parent

    return accumulated_feedback[::-1], accumulated_reflection[::-1]


SEED_FILE = "task_id.json"


def load_seed():
    """Load the current seed from file.
    
    Returns:
        Current seed value, or 0 if file doesn't exist
    """
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, 'r') as f:
            return json.load(f)["current_seed"]
    return 0


def save_seed(seed):
    """Save the current seed to file.
    
    Args:
        seed: Seed value to save
    """
    with open(SEED_FILE, 'w') as f:
        json.dump({"current_seed": seed}, f)
def run_mcts(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    n: int = 4,
    number_of_tests: int = 2
) -> None:
    """Run MCTS-based code generation with EMCS evaluation.
    
    Args:
        dataset: List of programming tasks
        model_name: Name of the LLM to use
        language: Programming language ('py', 'rs', 'go')
        max_iters: Maximum MCTS iterations
        pass_at_k: Pass@k metric parameter
        log_path: Path to save results
        verbose: Whether to print verbose logs
        is_leetcode: Whether running on LeetCode benchmark
        n: Number of child nodes to expand per iteration
        number_of_tests: Number of internal tests to generate
    """
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    test_model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    cur_func_impl = None
    len_task = 0
    
    for idx, item in enumerate(dataset):
        print(f"{'='*50} Task {idx} {'='*50}")
        logging.info(f"Processing task {idx}")


        if idx >= 0:
            len_task += 1

            if language == 'py':
                pattern = r'def\s+(\w+)\s*\('
                match = re.search(pattern, item["prompt"])
                method_name = match.group(1) if match else None
            else:
                pattern = r'fn\s+(\w+)\s*\('
                match = re.search(pattern, item["prompt"])
                method_name = match.group(1) if match else None

            cur_func_impl = None
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                test_fewshot = item["test"]

                if language == 'py':
                    test_list = [test.strip() for test in test_fewshot.splitlines() if "assert" in test]
                    pattern = r'assert\s+([a-zA-Z_]\w*)\('
                    match = re.search(pattern, test_list[0])
                    if match is not None:
                        assert_name = match.group(1)
                        tests_i = [
                            case.replace(assert_name, method_name) if assert_name in case else case 
                            for case in test_list
                        ]
                else:
                    test_list = [test.strip() for test in test_fewshot.splitlines() if "assert_eq!" in test]
                    method_pattern = r'assert_eq!\s*\(\s*([a-zA-Z_]\w*)\('
                    match = re.search(method_pattern, test_list[0])
                    if match is not None:
                        assert_name = match.group(1)
                        tests_i = [
                            case.replace(assert_name, method_name) if assert_name in case else case 
                            for case in test_list
                        ]

            print(f"Test cases: {tests_i}")
            logging.info(f"Test cases: {tests_i}")
            while cur_func_impl is None:
                cur_func_impl_list = gen.func_impl(item["prompt"], model, "simple")
                cur_func_impl = cur_func_impl_list[0]
            logging.info(f"Initial solution generated: {cur_func_impl_list}")
            root = Node(cur_func_impl)
            root.model = cur_func_impl_list[1]
            if isinstance(cur_func_impl, str):
                if language == 'py':
                    pattern = r'def\s+(\w+)\s*\('
                else:
                    pattern = r'fn\s+(\w+)\s*\('
                match = re.search(pattern, cur_func_impl)
                if match is not None:
                    cur_method_name = match.group(1)
                    if cur_method_name is not None and cur_method_name != method_name:
                        cur_func_impl = cur_func_impl.replace(cur_method_name, method_name)
            reflections = []
            implementations = []
            test_feedback = []
            is_solved = False

            implementations.append(cur_func_impl)
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
            test_feedback.append(feedback)

            if is_passing:
                print(f"First attempt succeeded: {feedback}")
                from run import llm_manager
                llm_manager.update_reward(cur_func_impl_list[1], 1)
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10
                )
                is_solved = is_passing
                num_success += int(is_passing)
                item["acc"] = round(num_success / len_task, 2)
                write_jsonl(log_path, [item], append=True)
                print(f"Success count: {num_success}")
                print_v(f'Completed {idx+1}/{num_items}: acc = {round(num_success / len_task, 2)}')
                continue

            print(f"Test feedback: {feedback}")

            reflection = gen.self_reflection(cur_func_impl, feedback, model)
            reflections += [reflection]
            root.test_feedback = feedback
            root.reflection = reflection

            for cur_iter in range(max_iters):
                node = root
                trajectory = {
                    'solutions': [],
                    'feedbacks': []
                }

                while node.children:
                    node = node.best_child()
                    trajectory['solutions'].append(node.solution)

                for _ in range(n):
                    new_solution = None
                    strategy = "mcts"
                    prev_func_impl = node.solution
                    feedback = node.test_feedback
                    reflection = node.reflection
                    if reflection is None:
                        reflection = ""
                    acc_feedback, acc_reflection = gather_context_from_tree(node)

                    while new_solution is None:
                        new_solution_list = gen.func_impl(
                            func_sig=item["prompt"],
                            model=model,
                            strategy=strategy,
                            prev_func_impl=prev_func_impl,
                            feedback=feedback,
                            self_reflection=reflection,
                            acc_feedback = acc_feedback,
                            acc_reflection = acc_reflection
                        )
                        new_solution = new_solution_list[0]
                        if isinstance(new_solution, str):
                            if language == 'py':
                                pattern = r'def\s+(\w+)\s*\('
                            else:
                                pattern = r'fn\s+(\w+)\s*\('

                            match = re.search(pattern, new_solution)
                            if match is not None:
                                new_method_name = match.group(1)
                                if new_method_name is not None and new_method_name != method_name:
                                    new_solution = new_solution.replace(new_method_name, method_name)

                    combined_context = "\nPrevious Trial\n\n" + new_solution
                    child = Node(new_solution, parent=node, context=combined_context, depth=node.depth + 1)
                    child.model = new_solution_list[1]
                    node.children.append(child)

                    reward_real = 0
                    for child in node.children:
                        is_passing_internal, feedback_internal, _ = exe.execute(child.solution, tests_i)
                        if not is_passing_internal:
                            reflection = gen.self_reflection(child.solution, feedback_internal, model)
                            reflections.append(reflection)
                            child.reflection = reflection
                            child.test_feedback = feedback_internal
                            child.context += (
                                "\n\nPrevious Trial\n\n" + child.solution + 
                                "\n\nTest results: \n" + feedback_internal + 
                                "\n\nSelf-reflection: " + reflection
                            )
                        else:
                            print(f"Tests passed: {feedback_internal}")
                            child.context += (
                                "\n\nPrevious Trial\n\n" + child.solution + 
                                "\n\nTest results: \n" + feedback_internal
                            )
                            child.reflection = ""
                            child.test_feedback = feedback_internal

                        if "Tested passed:" in feedback_internal:
                            passed_section = feedback_internal.split("Tests failed:")[0]
                            reward_internal = len([
                                line for line in passed_section.split("Tested passed:")[1].splitlines() 
                                if line.strip() != ''
                            ])
                            if len(tests_i) > 0:
                                reward_internal = reward_internal / len(tests_i)
                            else:
                                reward_internal = reward_internal
                                print("Warning: tests_i is empty")
                        else:
                            reward_internal = 0
                        if is_passing_internal or cur_iter == max_iters - 1:
                            is_passing = exe.evaluate(
                                item["entry_point"], child.solution, item["test"], timeout=10
                            )
                            if is_passing:
                                print(f"Solution found! Entry point: {item['entry_point']}")
                                item["solution"] = child.solution
                                is_solved = True
                                reward_real = 1
                            break

                    if is_solved:
                        llm_manager.update_reward(child.model, 1)
                        break

                    print(f"Reward internal: {reward_internal}, Reward real: {reward_real}")
                    reward = reward_internal + reward_real
                    child.update(reward)
                    llm_manager.update_reward(child.model, reward)
                    
                    temp = child
                    while temp.parent:
                        temp = temp.parent
                        temp.update(reward)

                if is_solved:
                    break
            if is_solved:
                best_solution = item["solution"]
            else:
                best_solution = root.best_child_value().solution
                item["solution"] = best_solution

            is_passing, cur_feedback, _ = exe.execute(new_solution, tests_i)
            test_feedback.append(cur_feedback)
            is_passing = exe.evaluate(item["entry_point"], best_solution, item["test"], timeout=10)
            if is_passing:
                num_success += 1

            reflections.append("MCTS reflections")
            implementations.append(best_solution)
            item["is_solved"] = is_passing
            item["reflections"] = reflections
            item["implementations"] = implementations
            item["test_feedback"] = test_feedback
            item["acc"] = round(num_success / len_task, 2)
            write_jsonl(log_path, [item], append=True)

            print_v(f'Completed {idx+1}/{num_items}: acc = {round(num_success / len_task, 2)}')
