import os
import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType

load_dotenv()

trace_provider = register(
    project_name="excel-agent",
    project_type=ProjectType.OBSERVE,
    session_name="excel-agent-session",
)


# Available data files mapping
AVAILABLE_DATA_FILES = {
    'sales_q1': 'sales_q1.csv',
    'sales_q2': 'sales_q2.csv', 
    'customers': 'customers_sample.csv',
    'customer' : 'customers_sample.csv',
    'q1': 'sales_q1.csv',  
    'q2': 'sales_q2.csv'
}

def load_sample_data(filepath):
    """Load a CSV file containing sample data"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Warning: Couldn't load sample data from {filepath}: {e}")
        return None

def get_table_data(table_name):
    """Get data from the specified table name"""
    if table_name.lower() in AVAILABLE_DATA_FILES:
        file_path = AVAILABLE_DATA_FILES[table_name.lower()]
        data = load_sample_data(file_path)
        return data, file_path
    return None, None

# --- Tool Calling Functions ---

def set_formula_cell_value(sheet_name, cell_reference, formula):
    """
    Tool function to simulate setting a formula in a specific cell.
    In a real implementation, this might interact with an Excel API.
    
    Args:
        sheet_name: Name of the sheet to modify
        cell_reference: Cell reference (e.g., 'A1', 'B2:C3')
        formula: Excel formula to set in the cell
    
    Returns:
        A string confirming the action
    """
    # This is a simulation - in a real app, this would modify an actual spreadsheet
    print(f"\n[Tool Call] Setting formula in {sheet_name}!{cell_reference}:")
    print(f"==> {formula}")
    return f"Formula set in {sheet_name}!{cell_reference}"

def execute_apps_script(code_string, action_title="Execute Apps Script"):
    """
    Tool function to simulate executing a Google Apps Script.
    In a real implementation, this might call the Google Apps Script API.
    
    Args:
        code_string: JavaScript code to execute
        action_title: Title/description of the action
    
    Returns:
        A string confirming the action
    """
    # This is a simulation - in a real app, this would execute the script
    print(f"\n[Tool Call] {action_title}:")
    print(f"==> Running script: {code_string[:100]}{'...' if len(code_string) > 100 else ''}")
    return f"Executed: {action_title}"

# --- Hardcoded Examples ---

# Predefined examples - just queries
HARDCODED_EXAMPLES = [
    "Calculate the total sales amount in the sales_q1 table",
    "Find the average order value across all records in sales_q2",
    "Count how many unique customers we have in the customers table"
]

# --- OpenAI Integration ---

# Initialize the OpenAI client if API key is available
client = None
try:
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
    else:
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI features will be disabled.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("OpenAI features will be disabled.")

def identify_table_name(user_query):
    """Use LLM to identify which table the user is referring to in their query"""
    if not client:
        return None, "ERROR: OpenAI client not initialized."

    available_tables = ", ".join(AVAILABLE_DATA_FILES.keys())
    
    system_prompt = (
        "You are an assistant that identifies which table or dataset a user is referring to in a query. "
        f"Available tables are: {available_tables}. "
        "ONLY return the exact table name you identified, or 'NONE' if no table is mentioned. "
        "Do not provide explanations or additional text."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Using a smaller model for this simple task
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Identify the table name in this query: '{user_query}'"}
            ],
            temperature=0,
            max_tokens=50
        )
        table_name = response.choices[0].message.content.strip().lower()
        
        # Check if the response is valid
        if table_name == 'none':
            return None, "ERROR: No table name was mentioned in your query. Please specify which table you want to analyze."
        
        if table_name in AVAILABLE_DATA_FILES:
            return table_name, None
        else:
            return None, f"ERROR: The identified table '{table_name}' is not available. Available tables are: {available_tables}"
            
    except Exception as e:
        return None, f"ERROR: Could not identify table name: {e}"

def generate_excel_formula(user_query, table_data, table_name):
    """Generate an Excel formula using OpenAI based on the user query and specific table data"""
    if not client:
        return "ERROR: OpenAI client not initialized. Cannot generate formula."

    if table_data is None:
        return "ERROR: No data available for the specified table."
    
    # Prepare data context information
    columns = ", ".join(table_data.columns.tolist())
    row_count = len(table_data)
    sample_data = table_data.head(3).to_dict()
    
    data_context = f"""
Table name: {table_name}
File: {AVAILABLE_DATA_FILES.get(table_name, 'Unknown')}
Columns: {columns}
Row count: {row_count}
Sample data (first 3 rows): {sample_data}
"""
    
    # Create the system and user prompts
    system_prompt = (
        "You are an expert in Excel and Google Sheets formulas. "
        "Given a user query about data analysis and a specific dataset, generate the most appropriate Excel/Sheets formula. "
        "Use only standard Excel/Google Sheets functions that work in both platforms. "
        "Return ONLY the formula with no explanations or markdown. Always include the table name in your formula."
    )
    
    user_prompt = (
        f"Data context:\n{data_context}\n\n"
        f"User query: {user_query}\n\n"
        f"Respond with ONLY the Excel formula:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        formula = response.choices[0].message.content.strip()
        
        # Simple validation - ensure it looks like a formula
        if not formula.startswith('='):
            formula = f"={formula}"
            
        return formula
    except Exception as e:
        return f"ERROR: Could not generate formula: {e}"

# --- Agent Logic ---

def process_query(user_query):
    """Process the user query through the complete workflow to return an Excel formula"""
    # Step 1: Identify which table the user is referring to
    print("Step 1: Identifying table name mentioned in query...")
    table_name, error = identify_table_name(user_query)
    
    if error:
        print(f"Error identifying table: {error}")
        return error
    
    print(f"Identified table: {table_name}")
    
    # Step 2: Load the specific table data
    print(f"Step 2: Loading data for table '{table_name}'...")
    table_data, file_path = get_table_data(table_name)
    
    if table_data is None:
        error_msg = f"ERROR: Could not load data for table '{table_name}'"
        print(error_msg)
        return error_msg
    
    print(f"Loaded {len(table_data)} rows from {file_path}")
    
    # Step 3: Generate the Excel formula
    print("Step 3: Generating Excel formula...")
    formula = generate_excel_formula(user_query, table_data, table_name)
    
    # Step 4: Simulate setting the formula (if valid)
    if not formula.startswith("ERROR:"):
        set_formula_cell_value("Results", "A1", formula)
    
    return formula

# --- Command Line Interface ---

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Excel Formula Generator Agent')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode (continuous prompt processing)')
    return parser.parse_args()

# --- Modes of Operation ---

def run_hardcoded_mode():
    """Run the hardcoded examples mode."""
    print("\n--- Excel Formula Generator (Hardcoded Mode) ---")
    print("Running predetermined examples:")
    
    for i, example_query in enumerate(HARDCODED_EXAMPLES, 1):
        print(f"\nExample {i}: {example_query}")
        formula = process_query(example_query)
        
        if formula.startswith("ERROR:"):
            print(f"❌ {formula}")
        else:
            print(f"✅ Formula: {formula}")
    
    print("\nHardcoded examples completed.")

def run_interactive_mode():
    """Run the interactive mode."""
    print("\n--- Excel Formula Generator (Interactive Mode) ---")
    print("Enter your data analysis query (or type 'quit' to exit).")
    print(f"Available tables: {', '.join(AVAILABLE_DATA_FILES.keys())}")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            if not user_input.strip():
                continue
                
            formula = process_query(user_input)
            
            if formula.startswith("ERROR:"):
                print(f"\n❌ {formula}")
            else:
                print(f"\n✅ Generated Formula: {formula}")
            
        except EOFError:
            # Handle Ctrl+D
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            break

    print("\nExiting Excel Formula Generator.")

# --- Main Execution ---

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    if args.interactive:
        # Run in interactive mode
        run_interactive_mode()
    else:
        # Run hardcoded examples mode
        run_hardcoded_mode() 