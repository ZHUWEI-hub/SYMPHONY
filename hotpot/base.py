"""Base task class for different benchmarks."""

import os


class Task:
    """Base class for tasks.
    
    Subclasses should implement:
    - __len__: Return the number of tasks
    - get_input: Get input for a specific task index
    - test_output: Evaluate the output for a specific task
    """
    
    def __init__(self):
        pass

    def __len__(self) -> int:
        """Return the number of tasks in the dataset."""
        pass

    def get_input(self, idx: int) -> str:
        """Get the input for the task at the given index."""
        pass

    def test_output(self, idx: int, output: str):
        """Test and evaluate the output for the task at the given index."""
        pass