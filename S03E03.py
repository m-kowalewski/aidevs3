import json
import os
import re
from loguru import logger
from typing import List

from utils_ai import aidevs_send_answer, aidevs_s03e03_query, openai_create
from utils_files_and_text import extract_answer


def format_column_info_from_string(data: str) -> str:
    """
    Parses and formats column information from a JSON string containing SQL table definitions.

    Args:
        data (str): JSON string containing table information.

    Returns:
        str: A nicely formatted string about the columns in the table.
    """
    # Parse the JSON string
    parsed_data = json.loads(data)
    table_info = parsed_data.get("reply", [])[0]

    # Extract table name and "Create Table" statement
    table_name = table_info.get("Table", "Unknown Table")
    create_table_stmt = table_info.get("Create Table", "")

    # Extract column definitions using regex
    column_pattern = (
        r"`(?P<column_name>\w+)`\s+(?P<type>[\w\(\)]+)(?:\s+(?P<extra>.*?))?,?"
    )
    columns = re.findall(column_pattern, create_table_stmt)

    # Prepare the formatted output
    output = [f"Table: {table_name}", "Columns:"]
    for column_name, column_type, extra in columns:
        extra_info = f" ({extra.strip()})" if extra.strip() else ""
        output.append(f"  - {column_name}: {column_type}{extra_info}")

    return "\n".join(output)


def run():
    task: str = "które aktywne datacenter (DC_ID) są zarządzane przez pracowników, którzy są na urlopie (is_active=0)"
    system_template_table_selection: str = """
    ### Prompt:

    You are tasked with determining which tables are necessary to solve a user-provided task. The user will provide a task description and a list of available table names. Your job is to analyze the task and select the tables that are most relevant to completing it.
    
    ### Guidelines:
    1. **Analyze the task**: Carefully read the task description to understand what data is required and what operations need to be performed.
    2. **Review the table names**: Examine the list of available table names to identify those that might contain the relevant data for the task.
    3. **Think through the solution**:
       - Determine what data is required for the task and which table(s) are likely to contain it based on their names.
       - Consider if any relationships between tables might need to be used (e.g., joining tables for additional information).
    4. **Show your reasoning**: Before providing the answer, explain your thought process for selecting the necessary tables.
    5. **Provide your answer**: List the selected tables inside `<ANSWER></ANSWER>` tags.
    
    ### Example Input:
    ```
    Task: Retrieve the names and email addresses of customers who placed orders in the last month.  
    Table names: Customers, Orders, Products, Payments, Shipping
    ```
    
    ### Example Output:
    ```
    Reasoning:  
    - The task requires customer names and email addresses. This information is likely in the `Customers` table.  
    - We need to filter for customers who placed orders, so the `Orders` table will also be necessary to find recent orders.  
    - The other tables (`Products`, `Payments`, `Shipping`) are not directly related to retrieving customer contact information or order dates for this task.  
    
    <ANSWER>Customers, Orders</ANSWER>
    ```  
    
    ### Notes:
    - Focus on selecting only the tables directly relevant to the task to avoid unnecessary complexity.
    - Ensure the reasoning is clear, logical, and directly tied to the task description.
    """
    system_template_query_creation: str = """
    ### Prompt:
    You are tasked with writing SQL queries (string) based on the table structure provided by the user and the task description they give. Before providing the final SQL query, show your reasoning process to ensure the query is well thought out.
    
    ### Guidelines:
    1. **Understand the provided table structure**: Review the structure of the tables provided by the user. The structure includes the table names, column names, and data types.
    2. **Understand the user task**: Read the task description carefully to determine what needs to be queried from the database.
    3. **Think through the solution**:
       - Consider the necessary tables and how they relate to the task.
       - Think about any filters (`WHERE`), groupings (`GROUP BY`), joins (`JOIN`), or sorting (`ORDER BY`) needed to answer the task.
       - If more than one table is involved, consider how to join them appropriately using the correct key columns.
    4. **Provide your reasoning process**: Describe your thought process for determining the SQL query.
    5. **Write the SQL query**: After thinking through the solution, provide the SQL query wrapped in `<ANSWER></ANSWER>` tags.
    
    ### Example Input:
    ```
    <TABLE STRUCTURE>
    1. Employees
       - employee_id (int)
       - first_name (varchar)
       - last_name (varchar)
       - department (varchar)
       - email (varchar)
       - hire_date (date)
    
    2. Departments
       - department_id (int)
       - department_name (varchar)
    </TABLE STRUCTURE>
    
    <TASK>
    Get the names (first and last) of all employees who are in the "Sales" department and sort them by their last name.
    </TASK>
    ```
    
    ### Example Output:
    ```
    Reasoning:  
    - We need to retrieve the first and last names of employees. This information is in the `Employees` table.  
    - The user wants to filter for employees in the "Sales" department, which is specified in the `department` column of the `Employees` table.  
    - The result should be sorted by the last name, so we will use the `ORDER BY` clause on the `last_name` column.  
    - The `Departments` table is not necessary for this query because the department information is already stored in the `Employees` table.
    
    <ANSWER>
    SELECT first_name, last_name
    FROM Employees
    WHERE department = 'Sales'
    ORDER BY last_name;
    </ANSWER>
    ```  
    
    ### Notes:
    - The reasoning process helps ensure the query is logically sound before providing the final answer.
    - Make sure the query is tailored to the exact task, using only the necessary tables and columns.
    """
    query_show_tables: str = "show tables;"
    response_show_tables = aidevs_s03e03_query(query_show_tables)

    parsed_data = json.loads(response_show_tables.content.decode("utf-8"))
    table_names: List[str] = [
        entry["Tables_in_banan"] for entry in parsed_data["reply"]
    ]
    logger.debug(f"TABLE NAMES: {table_names}")

    human_template_table_selection: str = (
        f"Task: {task}\nTable names: {', '.join(table_names)}"
    )
    response_table_selection = openai_create(
        system_template_table_selection, human_template_table_selection, model="gpt-4o"
    )
    logger.debug(f"TABLES: {response_table_selection.content}")
    selected_tables: List[str] = extract_answer(response_table_selection.content).split(
        ", "
    )

    selected_tables_description = dict()
    for selected_table in selected_tables:
        query_show_create_table: str = f"show create table {selected_table};"
        response_show_create_table = aidevs_s03e03_query(query_show_create_table)
        selected_tables_description[selected_table] = format_column_info_from_string(
            response_show_create_table.text
        )
    human_template_query_creation: str = (
        f"<TABLE STRUCTURE>{'; '.join(f'{k}: {v}' for k, v in selected_tables_description.items())}</TABLE STRUCTURE>\n"
        f"<TASK>{task}</TASK>"
    )
    response_query_creation = openai_create(
        system_template_query_creation, human_template_query_creation, model="gpt-4o"
    )
    logger.debug(f"TABLES: {response_query_creation.content}")

    query_solution: str = extract_answer(response_query_creation.content)
    logger.debug(f"FINAL QUERY: {query_solution}")

    response_solution = aidevs_s03e03_query(query_solution)

    parsed_data = json.loads(response_solution.text)
    # TODO: delete hardcoded "dc_id"
    answer = [int(item["dc_id"]) for item in parsed_data["reply"]]

    task_response = aidevs_send_answer(os.getenv("S03E03_TASK_NAME"), answer)
    if task_response.status_code == 200:
        logger.success(f"Request successful! Response:, {task_response.content}")
    else:
        logger.warning(
            f"Request failed with status code: {task_response.status_code}"
            f"Response content: {task_response.content}"
        )


if __name__ == "__main__":
    run()
