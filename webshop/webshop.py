"""WebShop task implementation."""

import os
import re
import random
import logging

from transformers import BertTokenizer

from base import Task
from prompt import *
from models import gpt3, gpt, gpt4


# Load tokenizer for token counting
cache_dir = os.getenv("BERT_TOKENIZER_PATH", "bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained(cache_dir)

def get_token_length(text):
    """Calculate the token length of a text string."""
    return len(tokenizer.encode(text))


max_token_length = 15000


class WebShopTask(Task):
    """Task class for WebShop interactive shopping.
    
    This task involves navigating an online shopping environment to find
    and select products based on natural language instructions.
    """
    
    def __init__(self):
        """Initialize the WebShop task."""
        super().__init__()
        self.steps = 7
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}
        self.reflections = []
    
    def test_output(self, idx: int, output: str):
        """Evaluate the quality of an output using GPT scoring."""
        output = output.split('Action:\n')[-1]
        prompt = score_prompt + output
        score_outputs = gpt(prompt, n=5, model='gpt-4')
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
    def generate_self_reflection(z, question):
        """Generate self-reflections on failed trajectories.
        
        Args:
            z: List of failed trajectory dictionaries with 'trajectory' and 'r' keys
            question: The original question
            
        Returns:
            List of reflection mappings with questions, trajectories, and reflections
        """
        reflection_mapping = []
        trajectories = ""

        sampled_items = random.sample(z, min(3, len(z)))
        failed_trajectories = [
            item['trajectory'] + f"\nReward: {item['r']}\n" 
            for item in sampled_items 
            if isinstance(item, dict) and 'trajectory' in item and 'r' in item
        ]
        
        for traj in failed_trajectories:
            trajectories += traj
            reflect_prompt = reflection_prompt.format(trajectory=traj)
            reflection = gpt3(reflect_prompt, max_tokens=300)
            trajectories += "Reflection: " + reflection[0] + "\n"
            
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
        input_text = x + y
        trajectories = ""
        
        if reflection_mapping_list:
            count = 0
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = (reflection_mapping['trajectory'] + 
                                      "Reflection: " + reflection_mapping['reflection'] + "\n")
                trajectories += traj_with_reflection
                count += 1
                if count == 3:  # Use maximum 3 reflections
                    break
            print(f"Assembled {count} reflection trajectories")
            prompt = prompt1_feedback.format(trajectories=trajectories, input=input_text)
            return prompt
        else:
            return prompt1.format(input=input_text)


        
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        """Create a voting prompt with multiple candidate trajectories.
        
        Args:
            x: The question
            ys: List of candidate trajectories
            
        Returns:
            Formatted voting prompt
        """
        prompt = score_prompt + "\n" + x + "\n\n"
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        """Parse voting outputs to count votes for each candidate.
        
        Args:
            vote_outputs: List of vote output strings
            n_candidates: Number of candidates
            
        Returns:
            List of vote counts for each candidate
        """
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best trajectory is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'Vote pattern no match: {[vote_output]}')
        return vote_results
    
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
                score = int(traj['r'] * 10) / 2
                trajectory = traj['trajectory']
                split_trajectory = trajectory.split('Action: ')
                first_part = split_trajectory[0]

                # Remove the first 'Action' and corresponding 'Observation'
                remaining_parts = split_trajectory[2:]

                # Reconstruct the trajectory string
                new_trajectory = 'Action: '.join([first_part] + remaining_parts)
                traj['trajectory'] = new_trajectory
                failed_trajectories += f"{y}\n{traj}\nReflection: {ref['reflection']}\nThus the correctness score is {score}\n"
            
            inp = y + ""
            prompt = score_prompt_feedback.format(s="", c="", trajectories=failed_trajectories, input=inp)
        else:
            inp = y + ""
            prompt = score_prompt.format(s="", c="", input=inp)
            
        return prompt

    
    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str, default_score=8, default_confidence=1.0):
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

