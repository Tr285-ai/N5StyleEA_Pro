# ai_assistant/ai_service.py
import os
import openai
from typing import Dict, Any, Optional
import json
from pathlib import Path
import re

class AIService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        openai.api_key = self.api_key
        self.model = "gpt-4"  # or "gpt-3.5-turbo" for faster, more cost-effective responses

    async def generate_code(
        self,
        task: str,
        context: str = "",
        language: str = "python",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate code based on the given task and context.
        """
        try:
            system_prompt = (
                "You are an expert AI coding assistant specialized in trading systems. "
                "Generate clean, efficient, and well-documented code. "
                f"Use {language} unless specified otherwise."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}\n\nContext:\n{context}"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None
            )
            
            content = response.choices[0].message['content'].strip()
            return {
                'status': 'success',
                'code': self._extract_code_blocks(content),
                'explanation': self._remove_code_blocks(content)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    async def analyze_code(
        self,
        code: str,
        task: str = "Review this code",
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Analyze and review code for improvements, bugs, or optimizations.
        """
        try:
            system_prompt = (
                "You are an expert code reviewer. Analyze the following code and provide: "
                "1. A brief summary of what the code does\n"
                "2. Potential bugs or issues\n"
                "3. Performance optimizations\n"
                "4. Security concerns\n"
                "5. Code style and best practices\n\n"
                f"Language: {language}\n"
                "Be concise but thorough."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}\n\nCode:\n```{language}\n{code}\n```"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more focused analysis
                max_tokens=1500
            )
            
            return {
                'status': 'success',
                'analysis': response.choices[0].message['content'].strip()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    async def explain_code(
        self,
        code: str,
        language: str = "python",
        detail_level: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        Explain what the code does in detail.
        """
        try:
            detail = {
                "beginner": "Explain in simple terms for beginners",
                "intermediate": "Provide a detailed explanation",
                "advanced": "Provide an in-depth technical analysis"
            }.get(detail_level.lower(), "Provide a detailed explanation")

            messages = [
                {"role": "system", "content": "You are a helpful coding assistant that explains code clearly and accurately."},
                {"role": "user", "content": f"{detail} of this {language} code:\n\n```{language}\n{code}\n```"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            return {
                'status': 'success',
                'explanation': response.choices[0].message['content'].strip()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    @staticmethod
    def _extract_code_blocks(text: str) -> str:
        """Extract code blocks from markdown text."""
        code_blocks = re.findall(r'```(?:[a-z]*\n)?(.*?)```', text, re.DOTALL)
        return '\n\n'.join(block.strip() for block in code_blocks) if code_blocks else ""

    @staticmethod
    def _remove_code_blocks(text: str) -> str:
        """Remove code blocks from text, leaving only the explanation."""
        return re.sub(r'```[a-z]*\n.*?\n```', '', text, flags=re.DOTALL).strip()

    async def generate_test_cases(
        self,
        code: str,
        language: str = "python",
        framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Generate test cases for the given code.
        """
        try:
            messages = [
                {"role": "system", "content": "You are an expert in software testing."},
                {"role": "user", "content": (
                    f"Generate comprehensive test cases for the following {language} code "
                    f"using {framework}. Include edge cases and error cases.\n\n"
                    f"Code:\n```{language}\n{code}\n```"
                )}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message['content'].strip()
            return {
                'status': 'success',
                'test_cases': content,
                'code_blocks': self._extract_code_blocks(content)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    async def optimize_code(
        self,
        code: str,
        language: str = "python",
        optimization_goal: str = "performance"
    ) -> Dict[str, Any]:
        """
        Optimize the given code based on the specified goal.
        """
        try:
            goals = {
                "performance": "Optimize for maximum execution speed",
                "readability": "Optimize for maximum readability and maintainability",
                "memory": "Optimize for minimum memory usage",
                "all": "Optimize for a balance of performance, memory usage, and readability"
            }
            
            goal = goals.get(optimization_goal.lower(), optimization_goal)
            
            messages = [
                {"role": "system", "content": "You are an expert code optimizer."},
                {"role": "user", "content": (
                    f"{goal} for the following {language} code. "
                    "Provide the optimized code and explain the changes made.\n\n"
                    f"Code to optimize:\n```{language}\n{code}\n```"
                )}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message['content'].strip()
            return {
                'status': 'success',
                'optimized_code': self._extract_code_blocks(content),
                'explanation': self._remove_code_blocks(content)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }