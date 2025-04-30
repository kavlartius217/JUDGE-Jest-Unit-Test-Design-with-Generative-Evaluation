from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import FileReadTool, DirectoryReadTool

import os
os.environ['OPENAI_API_KEY'] = ##ADD KEY

from crewai import LLM
llm_openai_1 = LLM(model='gpt-4o-mini', temperature=0)

"The Team"

@CrewBase
class Generator:
    """This crew is responsible for the jest case generation and the mocking strategy"""

    @agent
    def directory_structure_agent(self) -> Agent:
        return Agent(
            role="Project Architecture Cartographer",
            goal="Create a comprehensive map of the project's structure and component relationships to facilitate effective code segmentation",
            backstory="I specialize in interpreting complex software architectures by analyzing directory structures, file relationships, and dependency patterns. With extensive experience mapping full-stack applications, I can identify the architectural patterns being used, distinguish between frontend and backend components, recognize testing frameworks, and understand how different parts of the application interconnect. My insights provide the foundation for effective code segmentation and testing strategies.",
            llm=llm_openai_1,
            tools=[DirectoryReadTool('/content/gcc-national-registry-dashboard-Dev_Branch'), FileReadTool()]
        )
    
    @task
    def directory_structure_task(self) -> Task:
        return Task(
            description="""
            Analyze the project's directory structure to create a comprehensive architectural map:
            
            1. Generate a hierarchical representation of the directory structure
            2. Classify directories and files by their purpose (frontend, backend, config, tests, etc.)
            3. Identify the technology stack based on file extensions and configuration files
            4. Map dependencies between components by analyzing import statements and package configs
            5. Locate authentication-related code and understand its integration points
            6. Identify database models and data access layers
            7. Recognize API endpoints and service boundaries
            8. Detect testing frameworks and existing test files
            
            Pay special attention to architectural patterns that reveal component boundaries, such as:
            - MVC/MVVM structures
            - Microservices vs monolithic organization
            - Component-based frontend architectures
            - Service layers and dependency injection patterns
            """,
            expected_output="""
            A detailed architectural map containing:
            
            1. Directory Structure: A hierarchical representation of folders and key files
            2. Component Classification: Identification of frontend, backend, and shared components
            3. Technology Stack: List of frameworks, languages, and key libraries in use
            4. Dependency Graph: Visualization of how components interconnect
            5. Authentication Flow: Identification of auth-related components and their relationships
            6. Data Layer: Database schemas, models, and data access components
            7. API Surface: Endpoints, controllers, and service boundaries
            8. Testing Context: Existing test structure and frameworks
            9. Segmentation Recommendations: Natural boundaries for code segmentation based on the architecture
            
            The output should be structured in a way that the Code Segmentation Agent can use to make informed decisions about logical separation of components for testing.
            """,
            agent=self.directory_structure_agent()
        )
    
    @agent
    def code_segmentation_agent(self) -> Agent:
        return Agent(
            role="Code Structure Analyst",
            goal="Break down source code into logical, isolated segments that can be independently tested",
            backstory="As a code architect specializing in software decomposition, I analyze complex codebases and identify logical boundaries. With years of experience in various programming paradigms, I can recognize patterns, understand dependencies, and isolate functional units regardless of programming language or framework. My expertise in full-stack applications helps me identify the natural divisions between components in both frontend and backend systems.",
            llm=llm_openai_1,
            tools=[FileReadTool('/content/gcc-national-registry-dashboard-Dev_Branch/server/src/controller/auth.js')]
        )
    
    @task
    def code_segmentation_task(self) -> Task:
        return Task(
            description="Analyze the provided source code and segment it into logical, testable units. For each segment, identify: (1) its primary function/purpose, (2) its inputs and outputs, (3) its dependencies on other code segments, and (4) potential edge cases or areas of concern. Pay special attention to authentication flows, database interactions, and API endpoints. Each segment should be isolated enough to test independently with appropriate mocks.",
            expected_output="A structured list of code segments, each containing: (1) the segment name/identifier, (2) the location in the source code, (3) a brief functional description, (4) input parameters and expected outputs, (5) dependencies that will need mocking, and (6) recommendations for testing focus areas.",
            agent=self.code_segmentation_agent()
        )

    @agent
    def mock_generator_agent(self) -> Agent:
        return Agent(
            role="Dependency Isolation Specialist",
            goal="Create comprehensive mock objects and test fixtures that simulate real-world interactions for all external dependencies",
            backstory="I've specialized in creating realistic test environments for complex applications. With deep knowledge of mocking frameworks and test doubles, I can simulate databases, authentication systems, APIs, and other external dependencies with precision. My expertise allows for testing components in isolation while maintaining realistic behavior of their dependencies. I'm particularly skilled at mocking security contexts and authentication flows in full-stack applications.",
            llm=llm_openai_1
        )

    @task
    def mock_generator_task(self) -> Task:
        return Task(
            description="""
            For each code segment identified by the Code Structure Analyst, create appropriate mocks and test fixtures:
            
            1. Identify all external dependencies (databases, APIs, services, etc.)
            2. Create mock objects that simulate expected behavior of these dependencies
            3. Generate realistic test data that covers various scenarios including:
               - Authentication states (logged in, logged out, different permission levels)
               - Database responses (found records, empty results, errors)
               - API responses (success, failure, timeout, malformed responses)
               - User input variations (valid, invalid, edge cases)
            4. Ensure mocks maintain the contract expected by the code segment
            5. Create scenarios for happy paths and error conditions
            6. For authentication components, create test fixtures representing different user roles and permissions
            
            Focus on creating mocks that are:
            - Realistic enough to test real behaviors
            - Controllable to simulate specific conditions
            - Verifiable to confirm interactions occurred correctly
            """,
            expected_output="""
            For each code segment, provide:
            
            1. Mock Objects:
               - Mock definition for each external dependency
               - Configuration settings to control mock behavior
               - Verification points to assert correct interaction
               
            2. Test Fixtures:
               - Sample data representing different test scenarios
               - User contexts for authentication testing
               - Environment configurations for different test conditions
               
            3. Scenario Matrix:
               - Mapping of which mocks and fixtures apply to which test scenarios
               - Expected outcomes for each scenario
               
            The output should be structured to directly feed into the Test Case Generator's process.
            """,
            agent=self.mock_generator_agent(),
            context=[self.directory_structure_task(),self.code_segmentation_task()]
        )
    
    @agent
    def test_case_generator_agent(self) -> Agent:
        return Agent(
            role="Test Scenario Architect",
            goal="Create comprehensive test cases that verify functionality, edge cases, and error handling for each code segment",
            backstory="I'm an expert in designing test scenarios that reveal potential issues in code. With extensive experience in test-driven development and behavior-driven design, I craft tests that not only verify functionality but also document the expected behavior of systems. I specialize in full-stack application testing, with particular attention to user flows, data validation, and security considerations. My test cases balance thoroughness with practicality, ensuring critical paths are well-tested without creating excessive maintenance burden.",
            llm=llm_openai_1
        )

    @task
    def test_case_generator_task(self) -> Task:
        return Task(
            description="""
            Using the code segments from the Code Structure Analyst and the mocks from the Dependency Isolation Specialist, create comprehensive test cases:
            
            1. For each code segment, develop a suite of tests covering:
               - Basic functionality (happy path)
               - Input validation and boundary conditions
               - Error handling and edge cases
               - Security considerations
               - Performance concerns (where relevant)
               
            2. Each test case should specify:
               - Preconditions and setup requirements
               - Test inputs and parameters
               - Which mocks and fixtures to use
               - Expected outcomes and assertions
               - Cleanup actions if needed
               
            3. For authentication-related functionality:
               - Test cases for login, logout, session management
               - Permission verification and access control
               - Token handling and security measures
               
            4. For data handling components:
               - Data validation test cases
               - CRUD operation verification
               - Data transformation tests
               
            5. For user interfaces:
               - Input handling and form validation
               - UI state management tests
               - User flow scenarios
            """,
            expected_output="""
            A structured set of test cases for each code segment, including:
            
            1. Test Suite Structure:
               - Logical grouping of tests by functionality
               - Setup and teardown procedures
               
            2. Individual Test Cases:
               - Test name and description
               - Preconditions and environment setup
               - Test input data and parameters
               - Mock configuration and behavior
               - Step-by-step execution process
               - Expected outcomes and assertions
               - Edge cases and variations
               
            3. Coverage Analysis:
               - Assessment of test coverage for each code segment
               - Identification of untested or under-tested paths
               
            4. Testing Recommendations:
               - Prioritized list of test cases by importance
               - Suggestions for additional tests that may be valuable
               
            The output should be ready for evaluation by the Test Case Validator.
            """,
            agent=self.test_case_generator_agent(),
            context=[self.code_segmentation_task(),
                self.mock_generator_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.directory_structure_agent(),
                self.code_segmentation_agent(),
                self.mock_generator_agent(),
                self.test_case_generator_agent()
            ],
            tasks=[
                self.directory_structure_task(),
                self.code_segmentation_task(),
                self.mock_generator_task(),
                self.test_case_generator_task()
            ],
            process=Process.sequential,
            verbose=True
        )
