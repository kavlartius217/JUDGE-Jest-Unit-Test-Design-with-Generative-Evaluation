class Reasoning_tester(Flow[State]):
    model = 'gpt-4o-mini'
    g1 = Generator()
    "This an AI FeedBack Loop Tester"

    @start()
    def code_gen(self):
        response = self.g1.crew().kickoff({"feedback": ""})
        print(response.raw)
        self.state.code = response.raw

    @listen(or_(code_gen,"changes_made"))
    def code_review(self):
        response = completion(
            model=self.model,
            messages=[{
                "role": "system",
                "content": f"""You are an expert Jest Test Code Reviewer specialized in providing specific, actionable feedback for improving test quality. Your role is to carefully analyze generated Jest test files and provide detailed feedback on what corrections should be made Review this code {self.state.code}.

Your feedback should be precise, structured, and implementation-ready, focusing on:

1. CORRECTNESS ISSUES:
   - Syntax errors or Jest-specific implementation mistakes
   - Improper use of Jest API (describe, it, expect, beforeEach, etc.)
   - Incorrect mocking techniques or mock implementations
   - Test assertions that don't properly validate the expected behavior
   - Async/await usage problems in test cases

2. QUALITY IMPROVEMENTS:
   - Test coverage gaps for important code paths
   - Missing edge case or error handling tests
   - Insufficient validation of outputs or side effects
   - Opportunities for more robust assertions
   - Better test isolation and reduced test interdependence

3. BEST PRACTICES:
   - Consistent naming conventions for test suites and cases
   - Proper setup and teardown procedures
   - Appropriate use of Jest matchers (toBe vs toEqual vs toStrictEqual)
   - Mock cleanup and reset practices
   - Test readability and maintainability issues

4. IF THE CODE IS CORRECT:
   - Return a single string VALID

For each issue identified, provide:
- The exact location in the code (line number or code snippet)
- What is problematic or could be improved
- The recommended correction with sample code when applicable
- A brief explanation of why this change improves the test

Structure your feedback as a JSON object with the following format:
```json
{{
  "feedbackItems": [
    {{
      "type": "CORRECTNESS|QUALITY|BEST_PRACTICE",
      "location": "Describe where in the code (test suite, test case, line)",
      "issue": "Concise description of the problem",
      "recommendation": "Specific code change or addition needed",
      "explanation": "Why this change matters for test quality"
    }}
  ]
}}"""
            }]
        )
        print(response.choices[0].message.content)
        self.state.feedback = response.choices[0].message.content

    @router(code_review)
    def router(self):
        if self.state.feedback == 'VALID':
            print('proceed')
        else:
            print('make_changes')

    
    @listen('make_changes')
    def replay(self):
      task_id="4f3c4997-77ea-4387-ba86-35bf0f88336a"
      feedback={"feedback":self.state.feedback}
      response=self.g1.crew().replay(task_id=task_id,inputs=feedback)
      self.state.code=response
      print(response)
      return "changes_made"

    
    @listen("VALID")
    def show_code(self):
      print(self.state.code)

    
    
