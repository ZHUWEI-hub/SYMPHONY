"""DFS-based code generation strategy."""

import math
import sys
from typing import List, Dict, Tuple, Any

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

sys.set_int_max_str_digits(100000)


class Node:
    """Node in the DFS search tree."""
    
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

    def uct(self, exploration_weight=1.0):
        """Calculate UCT value."""
        if self.visits == 0:
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self):
        """Select the best child based on UCT values."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        """Select the best child based on value only."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.value)

    def update(self, reward: float):
        """Update visit count and value."""
        self.visits += 1
        self.value += reward
    

def prune_context_blocks(context: str, max_length: int) -> str:
    """Prune context by removing earliest trial blocks."""
    if len(context) <= max_length:
        return context
    
    blocks = context.split('Previous Trial')
    
    while len('trial'.join(blocks)) > max_length and blocks:
        blocks.pop(0)
    
    return 'trial'.join(blocks)


def gather_context_from_tree(node: Node) -> Tuple[List[str], List[str]]:
    """Gather feedback and reflections from node to root."""
    accumulated_feedback = []
    accumulated_reflection = []

    while node:
        if node.test_feedback:
            accumulated_feedback.append(node.test_feedback)
        if node.reflection:
            accumulated_reflection.append(node.reflection)
        node = node.parent

    return accumulated_feedback[::-1], accumulated_reflection[::-1]


def run_dfs(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    n: int = 5
) -> None:
    """Run DFS-based code generation.
    
    Args:
        dataset: List of programming tasks
        model_name: Name of the LLM to use
        language: Programming language ('py', 'rs', 'go')
        max_iters: Maximum iterations (not used in DFS)
        pass_at_k: Pass@k metric parameter
        log_path: Path to save results
        verbose: Whether to print verbose logs
        is_leetcode: Whether running on LeetCode benchmark
        n: Expansion parameter (not used in DFS)
    """

    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    max_depth = 8
    cur_func_impl = None

    for idx, item in enumerate(dataset):
        if is_leetcode:
            tests_i = item['visible_tests']
        else:
            tests_i = gen.internal_tests(item["prompt"], model, 6)

        while cur_func_impl is None:
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            
        root = Node(cur_func_impl)
        
        reflections = []
        implementations = []
        test_feedback = []
        is_solved = False
        it = 0

        implementations.append(cur_func_impl)
        assert isinstance(cur_func_impl, str)
        is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)
        test_feedback.append(feedback)

        if is_passing:
            is_passing = exe.evaluate(
                item["entry_point"], cur_func_impl, item["test"], timeout=10
            )
            is_solved = is_passing
            num_success += int(is_passing)
            item["acc"] = round(num_success / (idx + 1), 2)
            write_jsonl(log_path, [item], append=True)
            print_v(f'Completed {idx+1}/{num_items}: acc = {round(num_success / (idx + 1), 2)}')
            continue
        
        reflection = gen.self_reflection(cur_func_impl, feedback, model)
        reflections += [reflection]
        root.test_feedback = feedback
        root.reflection = reflection

        stack = [root]
        while stack and it < 50:
            node = stack.pop()

            if node.depth >= max_depth:
                continue

            new_solution = None
            strategy = "simple"
            prev_func_impl = node.solution
            feedback = node.test_feedback
            reflection = ""
            acc_feedback, acc_reflection = [], []

            while new_solution is None:
                new_solution = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="simple",
                    prev_func_impl=prev_func_impl,
                    feedback=feedback,
                    self_reflection=reflection,
                    acc_feedback = acc_feedback,
                    acc_reflection = acc_reflection
                )
            
            combined_context = "\nPrevious Trial\n\n" + new_solution
            child = Node(new_solution, parent=node, context=combined_context, depth=node.depth + 1)
            node.children.append(child)

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
                    reward_internal = reward_internal / len(tests_i)
                if is_passing:
                    best_solution = child.solution
                    break

            stack.append(child)
            it += 1

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
        item["acc"] = round(num_success / (idx + 1), 2)
        write_jsonl(log_path, [item], append=True)
        
        print_v(f'Completed {idx+1}/{num_items}: acc = {round(num_success / (idx + 1), 2)}')
