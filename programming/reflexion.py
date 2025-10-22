"""Reflexion strategy with self-reflection and iterative improvement."""

from typing import List

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    number_of_tests: int = 2
) -> None:
    """Run Reflexion strategy with self-reflection.
    
    Args:
        dataset: List of programming tasks
        model_name: Name of the LLM to use
        language: Programming language ('py', 'rs', 'go')
        max_iters: Maximum refinement iterations
        pass_at_k: Number of attempts for pass@k metric
        log_path: Path to save results
        verbose: Whether to print verbose logs
        is_leetcode: Whether running on LeetCode benchmark
        number_of_tests: Number of internal tests to generate
    """
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)
    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = None
        
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, number_of_tests)

            while cur_func_impl is None:
                cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
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
                break

            cur_iter = 1
            cur_feedback = feedback
            while cur_iter < max_iters:
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model)
                reflections += [reflection]

                prompt = item["prompt"]
                prompt += "\n[unit test results from previous impl]:\n"
                prompt += cur_feedback

                cur_func_impl = gen.func_impl(
                    func_sig=prompt,
                    model=model,
                    strategy="simple",
                    prev_func_impl=cur_func_impl,
                    feedback="",
                    self_reflection="",
                )
                implementations.append(cur_func_impl)
                assert isinstance(cur_func_impl, str)

                is_passing, cur_feedback, _ = exe.execute(cur_func_impl, tests_i)
                test_feedback.append(cur_feedback)

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

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        item["acc"] = round(num_success / (i + 1), 2)
        write_jsonl(log_path, [item], append=True)

        print_v(f'Completed {i+1}/{num_items}: acc = {round(num_success / (i + 1), 2)}')