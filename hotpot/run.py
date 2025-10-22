"""Main entry point for running HotPotQA experiments with SYMPHONY."""

import os
import json
import argparse
import logging
import math
from typing import List, Dict

from hotpotqa import HotPotQATask
from models import gpt_usage
from symphony import mcts_search
from tot import dfs_search
from rap import mcts_search


class AgentPool:
    """Multi-agent pool with UCB-based selection strategy."""

    def __init__(self, llm_names: List[str], exploration_weight: float = 20.0):
        """Initialize the agent pool.
        
        Args:
            llm_names: List of LLM names in the agent pool
            exploration_weight: Exploration parameter (alpha) for UCB algorithm
        """
        self.llm_names = llm_names
        self.exploration_weight = exploration_weight
        self.llm_stats = {name: {"total_reward": 0.0, "call_count": 0} for name in llm_names}
        self.total_calls = 0

    def select_llm(self) -> str:
        """Select next LLM using UCB (Upper Confidence Bound) algorithm.
        
        Returns:
            Name of the selected LLM
        """
        if self.total_calls < len(self.llm_names):
            for name in self.llm_names:
                if self.llm_stats[name]["call_count"] == 0:
                    return name

        ucb_values = {}
        for name, stats in self.llm_stats.items():
            if stats["call_count"] == 0:
                continue
            exploitation = stats["total_reward"] / stats["call_count"]
            exploration = math.sqrt(
                self.exploration_weight * math.log(self.total_calls) / (stats["call_count"] + 1)
            )
            ucb_values[name] = exploitation + exploration

        return max(ucb_values, key=ucb_values.get)

    def call_llm(self) -> str:
        """Call the UCB-selected LLM and update statistics.
        
        Returns:
            Name of the selected LLM
        """
        llm_name = self.select_llm()
        self.llm_stats[llm_name]["call_count"] += 1
        self.total_calls += 1
        return llm_name

    def update_reward(self, llm_name: str, reward: float) -> None:
        """Update the reward for a specific LLM.
        
        Args:
            llm_name: Name of the LLM
            reward: Reward value to add
            
        Raises:
            ValueError: If llm_name is not in the pool
        """
        if llm_name not in self.llm_names:
            raise ValueError(f"LLM {llm_name} not in agent pool")
        self.llm_stats[llm_name]["total_reward"] += reward

    def get_llm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all LLMs in the pool."""
        return self.llm_stats

    def get_average_reward(self, llm_name: str) -> float:
        """Get average reward for a specific LLM.
        
        Args:
            llm_name: Name of the LLM
            
        Returns:
            Average reward
            
        Raises:
            ValueError: If llm_name is not in the pool
        """
        if llm_name not in self.llm_names:
            raise ValueError(f"LLM {llm_name} not in agent pool")
        
        count = self.llm_stats[llm_name]["call_count"]
        if count == 0:
            return 0.0
        return self.llm_stats[llm_name]["total_reward"] / count

def run(args):
    """Run the HotPotQA experiment.
    
    Args:
        args: Argument object with experiment configuration
    """
    task = HotPotQATask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0

    log_dir = os.path.dirname(args.log)
    if log_dir:
        print(f"Log directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.abspath(args.log)
    print(f"Log file path: {log_path}")

    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info("Logging configuration successful.")

    count = 0
    task_accs = []
    info = []

    for i in range(args.task_start_index, args.task_end_index):
        if i >= 0:
            if args.algorithm == 'mcts':
                state, value, all_nodes, reward, em, question = mcts_search(args, task, i, args.iterations, True)
            elif args.algorithm == 'tot':
                state, value, all_nodes, reward, em = dfs_search(args, task, i, args.iterations)
            elif args.algorithm == 'rap':
                state, value, all_nodes, reward, em = mcts_search(args, task, i, args.iterations)
            else:
                raise ValueError(f"Unknown search algorithm: {args.algorithm}")

            if em is None:
                em = 0

            task_accs.append(em)
            cnt_avg = sum(task_accs) / len(task_accs)
            print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')

            result = (f"Task {i}: {question}\n"
                     f"Value: {value}, Reward: {reward}, EM: {em}\n"
                     f"Stats: len(task_accs)={len(task_accs)}, cnt_avg={cnt_avg}\n")

            log_dir = os.path.join(".", "log1")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "runname.txt")
            
            with open(log_file, "a") as file:
                file.write(result)
                file.write(f'Task {i} token usage: {gpt_usage()}\n')
                file.write("-" * 50 + "\n")
            
            print(f"Task {i} results written to file")
            print(f'Task {i} usage:', gpt_usage())
            logging.info(f'Task {i} usage: {gpt_usage()}')

       
    n = args.task_end_index - args.task_start_index
    print('Total usage:', gpt_usage())
    logging.info(f'Total usage: {gpt_usage()}')


class Args:
    """Simple argument container class for experiment configuration."""
    
    def __init__(
        self, 
        backend="Qwen2.5-7B-Instruct-1M", 
        temperature=1.0, 
        task_start_index=900, 
        task_end_index=1000, 
        prompt_sample='cot', 
        n_generate_sample=1, 
        n_evaluate_sample=1, 
        iterations=50, 
        log="", 
        algorithm="mcts"
    ):
        self.backend = backend
        self.temperature = temperature
        self.task_start_index = task_start_index
        self.task_end_index = task_end_index
        self.prompt_sample = prompt_sample
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.iterations = iterations
        self.log = log
        self.algorithm = algorithm

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' does not exist")


def get_args():
    """Get default arguments for the experiment.
    
    Returns:
        Args object with experiment configuration
    """
    args = Args(
        backend="Qwen2.5-7B-Instruct-1M",
        temperature=0.2,
        task_start_index=1,
        task_end_index=100,
        prompt_sample='cot',
        n_generate_sample=4,
        n_evaluate_sample=1,
        iterations=10,
        log="runname.log",
        algorithm="mcts"
    )
    return args


# Initialize the multi-agent pool with different LLMs
llm_manager = AgentPool([
    "Qwen2.5-7B-Instruct-1M", 
    "Mistral-7B-Instruct-v0.3", 
    "Meta-Llama-3.1-8B-Instruct"
])


if __name__ == '__main__':
    args = get_args()
    print(args)
    run(args)