"""HotPotQA task implementation."""

import os
import re
import random
import logging
from transformers import BertTokenizer

from base import Task
from hotpotPrompt import *
from models import gpt, gpt3


# Load tokenizer for token counting
cache_dir = os.getenv("BERT_TOKENIZER_PATH", "bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained(cache_dir)

def get_token_length(text):
    """Calculate the token length of a text string."""
    return len(tokenizer.encode(text))


max_token_length = 4000


class HotPotQATask(Task):
    """Task class for HotPotQA question answering.
    
    This task involves multi-hop question answering where the agent needs to
    search and reason over multiple Wikipedia pages to find the answer.
    """
    
    def __init__(self):
        """Initialize the HotPotQA task."""
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def test_output(self, idx: int, output: str):
        """Evaluate the quality of an output using GPT scoring."""
        output = output.split('Action:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=4, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            pattern = r".*correctness score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'Score pattern no match: {[score_output]}')
        print(scores)
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def generate_self_reflection(z, question):
        """Generate self-reflections on failed trajectories.
        
        Args:
            z: List of failed trajectories
            question: The original question
            
        Returns:
            List of reflection mappings with questions, trajectories, and reflections
        """
        reflection_mapping = []
        trajectories = ""

        sampled_items = random.sample(z, min(3, len(z)))
        failed_trajectories = "\n".join([f"{question}\n{traj}\n" for traj in z])
        failed_trajectories = [f"Question: {traj}" for traj in failed_trajectories.split("Question: ")[1:]]
        
        for traj in failed_trajectories:
            trajectories += traj
            
            reflect_prompt = reflection_prompt.format(trajectory=traj)
            
            reflection = gpt3(reflect_prompt, max_tokens=300)
            if len(reflection) <= 0:
                reflection = [""]
            trajectories += reflection[0] + "\n"
            
            reflection_mapping.append({
                'question': question,
                'trajectory': traj,
                'reflection': reflection[0]
            })

        return reflection_mapping

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[]):
        """Wrap the input with chain-of-thought prompt and reflections.
        
        Args:
            x: The question/input
            y: Current trajectory
            reflection_mapping_list: List of reflection mappings from failed attempts
            
        Returns:
            Formatted prompt string
        """
        question = x
        input_text = x + '\n' + y
        trajectories = ""
        
        if reflection_mapping_list:
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY \n\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                trajectories += traj_with_reflection
            
            prompt = cot_prompt_feedback.format(trajectories=trajectories, input=input_text)
            if get_token_length(prompt) > max_token_length:
                print("Prompt too long, using shortened version")
                trajectories = ""
                for reflection_mapping in reflection_mapping_list[:3]:
                    traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY \n\nReflection: " + reflection_mapping['reflection'] + "\n\n"
                    trajectories += traj_with_reflection
                prompt = cot_prompt_feedback_short.format(trajectories=trajectories, input=input_text)
            
            return prompt
        else:
            prompt = cot_prompt.format(input=input_text)
            if get_token_length(prompt) > max_token_length:
                prompt = cot_prompt_short.format(input=input_text)
            return prompt
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        """Create a voting prompt with multiple candidate trajectories.
        
        Args:
            x: The question
            ys: List of candidate trajectories
            
        Returns:
            Formatted voting prompt
        """
        prompt = vote_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best trajectory is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        
        # Extract the last Action for each trajectory
        last_actions = []
        for y in ys:
            # Split by line and reverse to start from the end
            lines = y.split('\n')[::-1]
            for line in lines:
                # Check for an Action line and get its content
                if "Action" in line:
                    last_actions.append(line.split('Action')[-1].strip(': '))
                    break

        assert len(last_actions) == 2, 'Expected to find 2 Actions'

        # Construct the prompt with the extracted Actions
        prompt = compare_prompt + f'Action 1:{last_actions[0]}\n\nAction 2:{last_actions[1]}\n'
        return prompt

    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        """Parse the comparison output to determine which trajectory is better.
        
        Args:
            compare_output: The comparison result string
            
        Returns:
            0 for trajectory 1, 1 for trajectory 2, 0.5 for equal, -1 for no match
        """
        if 'more correct trajectory is 1' in compare_output:
            return 0
        elif 'more correct trajectory is 2' in compare_output:
            return 1
        elif "two trajectories are similarly correct" in compare_output:
            return 0.5
        else:
            print(f'Compare output no match: {[compare_output]}')
            return -1
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        """Create a value evaluation prompt with optional failed trajectory feedback.
        
        Args:
            x: The question
            y: Current trajectory
            z: List of failed trajectories
            reflections: List of reflections on failed trajectories
            
        Returns:
            Formatted value evaluation prompt
        """
        question = x.split('\n')[0]
        
        if len(z) != 0:
            failed_trajectories = ""
            
            for traj, ref in zip(z, reflections):
                failed_trajectories += f"{question}\n{traj}\nThis trajectory is incorrect as {ref['reflection']}\nThus the correctness score is 1\n"
            
            inp = x + y + "\nThis trajectory is "
            
            prompt = value_prompt_reasoning_feedback.format(s="", c="", trajectories=failed_trajectories, input=inp)
            
            if get_token_length(prompt) > max_token_length:
                prompt = value_prompt_reasoning_feedback_short.format(s="", c="", trajectories=failed_trajectories, input=inp)
        else:
            inp = y + "\nThis trajectory is "
            prompt = value_prompt_reasoning.format(s="", c="", input=inp)
            
        return prompt

    
    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str, default_score=7, default_confidence=0.7):
        """Extract correctness score and confidence from evaluation output.
        
        Args:
            evaluate_prompt: The evaluation output string
            default_score: Default score if parsing fails
            default_confidence: Default confidence if parsing fails
            
        Returns:
            Tuple of (normalized_score, confidence)
        """
        evaluate_prompt = evaluate_prompt[0]

        numbers = re.findall(r'\d+\.?\d*', evaluate_prompt)

        int_values = []
        float_values = []
        for num in numbers:
            if '.' in num:
                float_values.append(float(num))
            else:
                int_values.append(int(num))

        correctness_score = default_score
        sorted_int_values = sorted(int_values, reverse=True)
        for num in sorted_int_values:
            if 1 <= num <= 10:
                correctness_score = num
                break

        confidence = default_confidence
        for num in float_values:
            if 0.0 <= num <= 1.0:
                confidence = num
                break
        if confidence == default_confidence:
            for num in int_values:
                if 0 <= num <= 1:
                    confidence = float(num)
                    break

        return correctness_score / 10, confidence
