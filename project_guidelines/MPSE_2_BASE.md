<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# given task at hand,  described solution, and a process of orchestrating different models with varied prompts responses length, chains of thought, prompts and voting procedures(rounds interleaved with various discussion variants), write code, all in python for such MVP solution.

```python
import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("math_ensemble.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class Agent:
    name: str
    model: str
    system_prompt: str
    temperature: float
    max_tokens: int
    
@dataclass
class Solution:
    agent_name: str
    answer: str
    explanation: str
    confidence: float
    
@dataclass
class VotingResult:
    answer: str
    confidence: float
    agents_in_agreement: List[str]
    
@dataclass
class MathProblemSolver:
    agents: List[Agent]
    voting_threshold: float = 0.7
    max_discussion_rounds: int = 2
    client: Any = None
    
    def __post_init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = openai.OpenAI(api_key=api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_solution(self, agent: Agent, problem: str, previous_solutions=None) -&gt; Solution:
        """Have an agent solve a math problem"""
        
        messages = [{"role": "system", "content": agent.system_prompt}]
        
        prompt = f"Problem: {problem}\n\nSolve this step by step. Be thorough in your analysis."
        
        if previous_solutions:
            prompt += "\n\nHere are solutions from other agents:\n\n"
            for solution in previous_solutions:
                prompt += f"{solution.agent_name}: {solution.answer}\n"
                prompt += f"Explanation: {solution.explanation[:200]}...\n\n"
            prompt += "\nConsider these solutions in your analysis, but provide your own approach."
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=agent.model,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens
            )
            
            explanation = response.choices[0].message.content
            
            # Extract answer - look for specific patterns
            answer = explanation
            if "answer:" in explanation.lower():
                answer_part = explanation.lower().split("answer:")
                answer = answer_part[1].strip().split("\n")[0]
            elif "therefore," in explanation.lower():
                answer_part = explanation.lower().split("therefore,")
                answer = answer_part[1].strip().split("\n")[0] 
            elif "conclusion:" in explanation.lower():
                answer_part = explanation.lower().split("conclusion:")
                answer = answer_part[1].strip().split("\n")[0]
            else:
                # Take the last paragraph as the answer
                paragraphs = explanation.split('\n\n')
                answer = paragraphs[-1].strip()
            
            # Calculate confidence based on model and temperature
            base_confidence = 0.8 if "gpt-4" in agent.model else 0.6
            confidence = base_confidence - (agent.temperature * 0.3)
            
            return Solution(
                agent_name=agent.name,
                answer=answer,
                explanation=explanation,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error with agent {agent.name}: {str(e)}")
            return Solution(
                agent_name=agent.name,
                answer="Error",
                explanation=f"Failed to compute: {str(e)}",
                confidence=0.0
            )
    
    def vote_on_solutions(self, solutions: List[Solution]) -&gt; VotingResult:
        """Conduct a vote among the solutions to find consensus"""
        
        # Group similar answers
        answer_groups = {}
        for solution in solutions:
            # Normalize answer for comparison
            normalized_answer = self._normalize_answer(solution.answer)
            
            if normalized_answer not in answer_groups:
                answer_groups[normalized_answer] = {
                    "agents": [],
                    "total_confidence": 0
                }
            
            answer_groups[normalized_answer]["agents"].append(solution.agent_name)
            answer_groups[normalized_answer]["total_confidence"] += solution.confidence
        
        # Find the answer with highest confidence
        best_answer = None
        best_confidence = 0
        best_agents = []
        
        for answer, data in answer_groups.items():
            if data["total_confidence"] &gt; best_confidence:
                best_confidence = data["total_confidence"]
                best_answer = answer
                best_agents = data["agents"]
        
        # Calculate overall confidence
        total_possible_confidence = sum(s.confidence for s in solutions)
        normalized_confidence = best_confidence / total_possible_confidence if total_possible_confidence &gt; 0 else 0
        
        # Find the original answer format from one of the solutions
        original_answer_format = next((s.answer for s in solutions 
                                      if self._normalize_answer(s.answer) == best_answer), best_answer)
        
        return VotingResult(
            answer=original_answer_format,
            confidence=normalized_confidence,
            agents_in_agreement=best_agents
        )
    
    def _normalize_answer(self, answer: str) -&gt; str:
        """Normalize an answer for comparison"""
        # Remove spaces, convert to lowercase
        normalized = answer.lower().replace(" ", "")
        
        # Remove common prefixes like "x="
        prefixes = ["x=", "y=", "z=", "answer:", "therefore,", "thus,", "hence,", "wefind", "theanswer"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        return normalized
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def facilitate_discussion(self, problem: str, solutions: List[Solution], voting_result: VotingResult) -&gt; str:
        """Generate a discussion among agents about the solutions"""
        
        system_prompt = """
        You are the facilitator of a discussion between several AI mathematical agents.
        Your job is to create a realistic dialogue between them as they analyze different approaches 
        to a math problem. Each agent should critique others' work constructively while defending their own.
        Focus on:
        1. Identifying mathematical errors or misconceptions
        2. Clarifying steps that might be unclear
        3. Suggesting alternative approaches
        4. Building consensus around the correct solution
        """
        
        prompt = f"""
        Problem: {problem}
        
        Current solutions:
        """
        
        for solution in solutions:
            prompt += f"\n{solution.agent_name}'s solution:\nAnswer: {solution.answer}\n"
            prompt += f"Explanation: {solution.explanation[:300]}...\n"
            prompt += f"Confidence: {solution.confidence:.2f}\n"
        
        prompt += f"""
        Current vote result:
        Best answer: {voting_result.answer}
        Confidence: {voting_result.confidence:.2f}
        Agents in agreement: {', '.join(voting_result.agents_in_agreement)}
        
        Generate a realistic discussion between these agents where they critique each others' approaches, 
        identify potential errors, and work together to find the correct solution. Each agent should maintain 
        their distinct personality and approach to problem-solving.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Discussion generation error: {str(e)}")
            return f"Discussion could not be generated due to an error: {str(e)}"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    def refine_solutions(self, problem: str, solutions: List[Solution], discussion: str) -&gt; List[Solution]:
        """Have agents refine their solutions based on the discussion"""
        
        refined_solutions = []
        
        for solution in solutions:
            agent = next((a for a in self.agents if a.name == solution.agent_name), None)
            if not agent:
                refined_solutions.append(solution)
                continue
            
            prompt = f"""
            Problem: {problem}
            
            Your previous solution (as {agent.name}):
            {solution.explanation}
            
            Discussion among agents:
            {discussion[:1500]} ... (discussion continues)
            
            Based on this discussion, please provide your refined solution to the problem.
            Consider the points raised by other agents and correct any mistakes in your approach.
            
            Format your response as:
            
            ANSWER: [Your clear, direct answer]
            
            EXPLANATION: [Your detailed step-by-step solution]
            
            CONFIDENCE: [A number between 0 and 1 representing your confidence]
            
            CHANGES: [Brief explanation of how your solution changed based on the discussion]
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=agent.model,
                    messages=[
                        {"role": "system", "content": agent.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens
                )
                
                refined_text = response.choices[0].message.content
                
                # Extract structured information
                refined_answer = "No clear answer provided"
                refined_explanation = refined_text
                refined_confidence = solution.confidence
                changes = "No changes specified"
                
                if "ANSWER:" in refined_text:
                    answer_parts = refined_text.split("ANSWER:")
                    if len(answer_parts) &gt; 1:
                        answer_text = answer_parts[1].split("\n")[0].strip()
                        if answer_text:
                            refined_answer = answer_text
                
                if "EXPLANATION:" in refined_text:
                    explanation_parts = refined_text.split("EXPLANATION:")
                    if len(explanation_parts) &gt; 1:
                        explanation_text = explanation_parts[1]
                        if "CONFIDENCE:" in explanation_text:
                            explanation_text = explanation_text.split("CONFIDENCE:")[0]
                        if "CHANGES:" in explanation_text:
                            explanation_text = explanation_text.split("CHANGES:")[0]
                        refined_explanation = explanation_text.strip()
                
                if "CONFIDENCE:" in refined_text:
                    confidence_parts = refined_text.split("CONFIDENCE:")
                    if len(confidence_parts) &gt; 1:
                        confidence_text = confidence_parts[1].split("\n")[0].strip()
                        try:
                            conf_value = float(confidence_text)
                            if 0 &lt;= conf_value &lt;= 1:
                                refined_confidence = conf_value
                        except ValueError:
                            pass
                
                if "CHANGES:" in refined_text:
                    changes_parts = refined_text.split("CHANGES:")
                    if len(changes_parts) &gt; 1:
                        changes = changes_parts[1].strip()
                
                refined_solutions.append(Solution(
                    agent_name=solution.agent_name,
                    answer=refined_answer,
                    explanation=refined_explanation,
                    confidence=refined_confidence
                ))
                
            except Exception as e:
                logger.error(f"Error refining solution for {agent.name}: {str(e)}")
                refined_solutions.append(solution)
        
        return refined_solutions
    
    def solve(self, problem: str) -&gt; Dict[str, Any]:
        """Main process to solve a math problem with ensemble of agents"""
        
        logger.info(f"Solving problem: {problem}")
        all_rounds_data = []
        
        # Initial solutions
        solutions = []
        for agent in self.agents:
            logger.info(f"Getting solution from {agent.name}")
            solution = self.get_solution(agent, problem)
            solutions.append(solution)
            logger.info(f"{agent.name} answer: {solution.answer}")
        
        # Initial voting
        voting_result = self.vote_on_solutions(solutions)
        logger.info(f"Initial vote: {voting_result.answer} with confidence {voting_result.confidence:.2f}")
        
        all_rounds_data.append({
            "round": 0,
            "solutions": [{"agent": s.agent_name, "answer": s.answer, "explanation": s.explanation} for s in solutions],
            "vote_result": {"answer": voting_result.answer, "confidence": voting_result.confidence}
        })
        
        # If initial confidence is high enough, return the result
        if voting_result.confidence &gt;= self.voting_threshold:
            logger.info("High confidence in initial vote, returning result")
            return {
                "problem": problem,
                "answer": voting_result.answer,
                "confidence": voting_result.confidence,
                "supporting_agents": voting_result.agents_in_agreement,
                "solutions": [{"agent": s.agent_name, "answer": s.answer, "explanation": s.explanation} for s in solutions],
                "rounds_data": all_rounds_data
            }
        
        # Discussion and refinement rounds
        current_solutions = solutions
        current_vote = voting_result
        
        for round_num in range(self.max_discussion_rounds):
            logger.info(f"Starting discussion round {round_num + 1}")
            
            # Generate discussion
            discussion = self.facilitate_discussion(problem, current_solutions, current_vote)
            
            # Refine solutions based on discussion
            refined_solutions = self.refine_solutions(problem, current_solutions, discussion)
            
            # Re-vote with refined solutions
            new_vote = self.vote_on_solutions(refined_solutions)
            logger.info(f"Round {round_num + 1} vote: {new_vote.answer} with confidence {new_vote.confidence:.2f}")
            
            all_rounds_data.append({
                "round": round_num + 1,
                "discussion": discussion,
                "solutions": [{"agent": s.agent_name, "answer": s.answer, "explanation": s.explanation} for s in refined_solutions],
                "vote_result": {"answer": new_vote.answer, "confidence": new_vote.confidence}
            })
            
            current_solutions = refined_solutions
            current_vote = new_vote
            
            # If confidence threshold is met, return the result
            if current_vote.confidence &gt;= self.voting_threshold:
                logger.info(f"Confidence threshold met in round {round_num + 1}")
                break
        
        # Return final result
        return {
            "problem": problem,
            "answer": current_vote.answer,
            "confidence": current_vote.confidence,
            "supporting_agents": current_vote.agents_in_agreement,
            "solutions": [{"agent": s.agent_name, "answer": s.answer, "explanation": s.explanation} 
                         for s in current_solutions],
            "rounds_data": all_rounds_data
        }

def init_math_ensemble():
    """Initialize the math problem solver with our ensemble of agents"""
    return MathProblemSolver(
        agents=[
            Agent(
                name="Formal Mathematician",
                model="gpt-4",
                system_prompt="You are a formal mathematician who solves problems with rigorous mathematical approaches. Always show your work in detail and use precise mathematical notation.",
                temperature=0.2,
                max_tokens=1000
            ),
            Agent(
                name="Intuitive Problem Solver",
                model="gpt-4",
                system_prompt="You are an intuitive problem solver who finds creative ways to approach math problems. You often see connections others miss and can find elegant solutions.",
                temperature=0.7,
                max_tokens=800
            ),
            Agent(
                name="Step-by-Step Teacher",
                model="gpt-4",
                system_prompt="You solve math problems by breaking them down into clear, sequential steps, as if teaching a student. You're meticulous and thorough in your explanations.",
                temperature=0.3,
                max_tokens=1200
            ),
            Agent(
                name="Verification Expert",
                model="gpt-4",
                system_prompt="You are an expert at verifying mathematical solutions. You're extremely careful about checking edge cases, units, and the validity of each step in a solution.",
                temperature=0.1,
                max_tokens=900
            ),
            Agent(
                name="Concise Analyst",
                model="gpt-3.5-turbo",
                system_prompt="You provide concise, efficient solutions to problems. You focus on the most direct approach and express your solution clearly and briefly.",
                temperature=0.4,
                max_tokens=700
            )
        ],
        voting_threshold=0.7,
        max_discussion_rounds=2
    )

def format_result_report(result: Dict[str, Any]) -&gt; str:
    """Format the result as a readable report"""
    report = [
        f"# Math Problem Solving Report",
        f"## Problem",
        f"{result['problem']}",
        f"## Final Answer",
        f"{result['answer']}",
        f"Confidence: {result['confidence']:.2f}",
        f"Supporting agents: {', '.join(result['supporting_agents'])}",
        f"## Solution Process"
    ]
    
    for round_data in result['rounds_data']:
        round_num = round_data['round']
        report.append(f"### Round {round_num}")
        
        if round_num &gt; 0 and 'discussion' in round_data:
            report.append("#### Discussion")
            report.append(round_data['discussion'])
        
        report.append("#### Solutions")
        for solution in round_data['solutions']:
            report.append(f"##### {solution['agent']}")
            report.append(f"Answer: {solution['answer']}")
            report.append(f"Explanation:\n{solution['explanation']}")
        
        report.append("#### Vote Result")
        report.append(f"Best answer: {round_data['vote_result']['answer']}")
        report.append(f"Confidence: {round_data['vote_result']['confidence']:.2f}")
    
    return "\n\n".join(report)

def main():
    # Initialize our ensemble solver
    solver = init_math_ensemble()
    
    # Test problem
    problem = input("Enter a math problem to solve: ")
    if not problem:
        problem = "Solve for x: 2x² - 8x + 6 = 0"
        print(f"Using example problem: {problem}")
    
    # Solve and get result
    result = solver.solve(problem)
    
    # Display result
    print("\n" + "="*50)
    print(f"Problem: {result['problem']}")
    print(f"Final Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Agents in agreement: {', '.join(result['supporting_agents'])}")
    print("="*50 + "\n")
    
    # Write result to files
    with open("math_problem_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    with open("math_problem_report.md", "w") as f:
        f.write(format_result_report(result))
    
    print("Results saved to math_problem_result.json and math_problem_report.md")

if __name__ == "__main__":
    main()
```


## Mathematics Problem-Solving Ensemble System

This code implements a multi-agent ensemble system for solving mathematical problems through collaborative reasoning, voting, and discussion. It's designed as an MVP submission for the Microsoft AI Agents Hackathon.

### Key Features

1. **Multiple Specialized Agents**: The system uses 5 different agents with distinct personalities and approaches:
    - Formal Mathematician (rigorous, precise)
    - Intuitive Problem Solver (creative approaches)
    - Step-by-Step Teacher (educational, thorough)
    - Verification Expert (checks edge cases)
    - Concise Analyst (direct, efficient)
2. **Sequential Problem-Solving Flow**:
    - Initial solutions generated independently by each agent
    - Voting to identify the most promising solution
    - Facilitated discussions where agents critique each other's work
    - Solution refinement based on discussions
    - Final voting to determine the best answer
3. **Robust Implementations**:
    - Answer normalization for accurate voting
    - Retry logic for API stability
    - Structured response parsing
    - Confidence estimation
    - Comprehensive logging

### How It Works

When executed, the system:

1. Takes a math problem as input
2. Has each agent solve it independently
3. Conducts a vote to find initial consensus
4. Facilitates discussions between agents about their approaches
5. Allows agents to refine their solutions based on the discussion
6. Repeats the voting process until confidence threshold is reached or max rounds completed
7. Produces both JSON data and a readable markdown report

### Usage

Simply run the script and enter a math problem when prompted. Results will be saved in two formats:

- `math_problem_result.json`: Complete data including all rounds and discussions
- `math_problem_report.md`: Human-readable report of the entire solution process

This demonstrates how multiple AI models can collaborate to solve problems more effectively than any single model alone.

