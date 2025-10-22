"""Python code executor."""

import ast
import astunparse

from .executor_utils import function_with_timeout
from typing import List
from .executor_types import ExecuteResult, Executor


class PyExecutor(Executor):
    """Executor for Python code."""
    
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        """Execute Python function with tests.
        
        Args:
            func: Function implementation code
            tests: List of assert statements
            timeout: Execution timeout in seconds
            
        Returns:
            ExecuteResult with test outcomes
        """
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        
        for i in range(num_tests):
            try:
                function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                success_tests += [tests[i]]
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # output: {output}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)

        feedback = "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"
            
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """Evaluate Python function with full test suite.
        
        Args:
            name: Function entry point name
            func: Function implementation code
            test: Full test suite code
            timeout: Execution timeout in seconds
            
        Returns:
            True if all tests pass, False otherwise
        """
        code = f"""{func}

{test}

check({name})
"""
        try:
            function_with_timeout(exec, (code, globals()), timeout)
            return True
        except Exception:
            return False


def get_call_str(assert_statement: str) -> str:
    """Extract function call from assert statement.
    
    Args:
        assert_statement: Assert statement string
        
    Returns:
        Function call string
    """
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left  # type: ignore
    except:
        call_str = ast_parsed.body[0].test  # type: ignore
    return astunparse.unparse(call_str).strip()


def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    """Get output of function call from assert statement.
    
    Args:
        func: Function implementation code
        assert_statement: Assert statement string
        timeout: Execution timeout in seconds
        
    Returns:
        String representation of output or error
    """
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
