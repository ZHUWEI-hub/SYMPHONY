"""Immediate refinement strategy with test feedback but no self-reflection."""

from typing import List

from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory


def run_immediate_refinement(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool,
) -> None:
    """Run immediate refinement strategy without self-reflection.
    
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
            tests_i = gen.internal_tests(item["prompt"], model, 1)

            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            assert isinstance(cur_func_impl, str)
            is_passing, feedback, _ = exe.execute(cur_func_impl, tests_i)

            if is_passing:
                is_passing = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10
                )
                is_solved = is_passing
                num_success += int(is_passing)
                break

            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection="No self-reflection"
                )
                assert isinstance(cur_func_impl, str)

                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)

                if is_passing or cur_iter == max_iters - 1:
                    is_passing = exe.evaluate(
                        item["entry_point"], cur_func_impl, item["test"], timeout=10
                    )
                    if is_passing:
                        item["solution"] = cur_func_impl
                        is_solved = True
                        num_success += 1
                    break

                cur_iter += 1
            cur_pass += 1

        is_solved = exe.evaluate(
            item["entry_point"], cur_func_impl, item["test"], timeout=10
        )

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["solution"] = cur_func_impl
        write_jsonl(log_path, [item], append=True)

        print_v(f'Completed {i+1}/{num_items}: acc = {round(num_success / (i + 1), 2)}')