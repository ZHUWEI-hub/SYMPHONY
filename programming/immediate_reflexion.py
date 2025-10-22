"""Immediate reflexion strategy without test execution."""

from typing import List

from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory


def run_immediate_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool
) -> None:
    """Run immediate reflexion strategy without executing tests.
    
    Args:
        dataset: List of programming tasks
        model_name: Name of the LLM to use
        language: Programming language ('py', 'rs', 'go')
        max_iters: Maximum refinement iterations
        pass_at_k: Number of attempts for pass@k metric
        log_path: Path to save results
        verbose: Whether to print verbose logs
        is_leetcode: Whether running on LeetCode benchmark
    """
    exe = executor_factory(language)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        cur_func_impl = ""
        
        while cur_pass < pass_at_k and not is_solved:
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            assert isinstance(cur_func_impl, str)

            cur_iter = 1
            feedback = "Test cases omitted"
            while cur_iter < max_iters:
                reflection = gen.self_reflection(cur_func_impl, feedback, model)
                reflections += [reflection]

                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=feedback,
                    self_reflection=reflection
                )
                assert isinstance(cur_func_impl, str)
                cur_iter += 1
            cur_pass += 1

        is_solved = exe.evaluate(
            item["entry_point"], cur_func_impl, item["test"], timeout=10
        )
        if is_solved:
            num_success += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["solution"] = cur_func_impl
        write_jsonl(log_path, [item], append=True)

        print_v(f'Completed {i+1}/{num_items}: acc = {round(num_success / (i + 1), 2)}')