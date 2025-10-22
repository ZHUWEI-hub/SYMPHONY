"""Main entry point for running code generation experiments with SYMPHONY."""

import os
import argparse
import logging
import math
from typing import List, Dict

from immediate_refinement import run_immediate_refinement
from immediate_reflexion import run_immediate_reflexion
from simple import run_simple
from reflexion import run_reflexion
from test_acc import run_test_acc
from utils import read_jsonl, read_jsonl_gz
from symphony import run_mcts
from dfs import run_dfs


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
class Args:
    """Simple argument container class for experiment configuration."""
    
    def __init__(
        self, 
        run_name, 
        root_dir="root", 
        dataset_path="root", 
        strategy="", 
        language="py", 
        model="", 
        pass_at_k=1,
        max_iters=10, 
        expansion_factor=3, 
        number_of_tests=6,
        is_leetcode=False, 
        verbose=False
    ):
        self.run_name = run_name
        self.root_dir = root_dir
        self.dataset_path = dataset_path
        self.strategy = strategy
        self.language = language
        self.model = model
        self.pass_at_k = pass_at_k
        self.max_iters = max_iters
        self.expansion_factor = expansion_factor
        self.number_of_tests = number_of_tests
        self.is_leetcode = is_leetcode
        self.verbose = verbose

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' does not exist")


def get_args():
    """Get default arguments for the experiment.
    
    Returns:
        Args object with experiment configuration
    """
    args = Args(
        run_name="5.18-3smallLLM-py-index",
        root_dir="root",
        dataset_path="./benchmarks/mbpp-py.jsonl",
        strategy="mcts",
        language="py",
        model="Mistral-7B-Instruct-v0.3",
        pass_at_k=1,
        max_iters=4,
        expansion_factor=3,
        number_of_tests=3,
        is_leetcode=False,
        verbose=True
    )
    return args


def strategy_factory(strategy: str):
    """Factory function to create strategy runners with appropriate parameters.
    
    Args:
        strategy: Name of the strategy to use
        
    Returns:
        Callable strategy runner function
        
    Raises:
        ValueError: If strategy is not supported
    """
    def kwargs_wrapper_gen(func, delete_keys=[]):
        """Generate a wrapper that removes specified keys from kwargs."""
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if strategy == "simple":
        return kwargs_wrapper_gen(run_simple, delete_keys=["expansion_factor", "max_iters"])
    elif strategy == "reflexion":
        return kwargs_wrapper_gen(run_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "mcts":
        return kwargs_wrapper_gen(run_mcts, delete_keys=["expansion_factor"])
    elif strategy == "dfs":
        return kwargs_wrapper_gen(run_dfs, delete_keys=["expansion_factor"])
    elif strategy == "immediate-reflexion":
        return kwargs_wrapper_gen(run_immediate_reflexion, delete_keys=["expansion_factor"])
    elif strategy == "immediate-refinement":
        return kwargs_wrapper_gen(run_immediate_refinement, delete_keys=["expansion_factor"])
    elif strategy == "test-acc":
        return kwargs_wrapper_gen(run_test_acc, delete_keys=["expansion_factor", "max_iters"])
    else:
        raise ValueError(f"Strategy `{strategy}` is not supported")


def main(args):
    """Run the code generation experiment.
    
    Args:
        args: Argument object with experiment configuration
    """
    logging.basicConfig(
        filename=os.path.join(args.root_dir, f"{args.run_name}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, 
        f"{dataset_name}_{args.strategy}_{args.max_iters}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}.jsonl"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    run_strategy = strategy_factory(args.strategy)

    if args.verbose:
        print(f"""
Starting run with the following parameters:
strategy: {args.strategy}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    print('Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    logging.info(f"Loaded {len(dataset)} examples")

    run_strategy(
        dataset=dataset,
        model_name=args.model,
        language=args.language,
        max_iters=args.max_iters,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        expansion_factor=args.expansion_factor,
        number_of_tests=args.number_of_tests,
        is_leetcode=args.is_leetcode
    )

    print(f"Done! Check out the logs in `{log_path}`")
    logging.info(f"Done! Check out the logs in `{log_path}`")


# Initialize the multi-agent pool with different LLMs
llm_manager = AgentPool([
    "Qwen2.5-7B-Instruct-1M", 
    "Mistral-7B-Instruct-v0.3", 
    "Meta-Llama-3.1-8B-Instruct"
])


if __name__ == "__main__":
    args = get_args()
    main(args)
