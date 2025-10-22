"""Abstract base class for code generators."""

from typing import List, Optional, Union
from abc import abstractmethod, ABC

from generators.model import ModelBase


class Generator(ABC):
    """Abstract base class for code generators.
    
    Subclasses must implement:
    - self_reflection: Generate self-reflection from code and feedback
    - func_impl: Generate code implementation
    - internal_tests: Generate internal test cases
    """
    
    @abstractmethod
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        """Generate self-reflection from code and feedback.
        
        Args:
            func: Function implementation code
            feedback: Test execution feedback
            model: LLM model to use
            
        Returns:
            Self-reflection text
        """
        ...

    @abstractmethod
    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        """Generate function implementation.
        
        Args:
            func_sig: Function signature/prompt
            model: LLM model to use
            strategy: Generation strategy ('simple', 'reflexion', etc.)
            prev_func_impl: Previous implementation (for reflexion)
            feedback: Test feedback (for reflexion)
            self_reflection: Self-reflection text (for reflexion)
            num_comps: Number of completions to generate
            temperature: Sampling temperature
            
        Returns:
            Generated code (single string or list of strings)
        """
        ...

    @abstractmethod
    def internal_tests(
        self,
        func_sig: str,
        model: ModelBase,
        max_num_tests: int = 5
    ) -> List[str]:
        """Generate internal test cases.
        
        Args:
            func_sig: Function signature/prompt
            model: LLM model to use
            max_num_tests: Maximum number of tests to generate
            
        Returns:
            List of test case strings
        """
        ...
