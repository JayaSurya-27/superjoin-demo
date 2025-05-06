import json
import os
from dotenv import load_dotenv
load_dotenv()

import argparse
from anyio import Path
import pandas as pd
from openai import OpenAI
from fi_instrumentation import register
from fi_instrumentation.fi_types import ProjectType, SpanAttributes, FiSpanKindValues
from excel_agent_eval_tags import eval_tags
from opentelemetry import trace



trace_provider = register(
    project_name="excel-agent-llm-eval",
    project_type=ProjectType.EXPERIMENT,
    eval_tags=eval_tags
)
trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer(__name__)




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
    with tracer.start_as_current_span(
        "load_sample_data",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.TOOL.value}
    ) as span:
        """Load a CSV file containing sample data"""
        try:
            span.set_attribute("file_path", filepath)
            span.set_attribute(SpanAttributes.RAW_INPUT, filepath)
            span.set_attribute(SpanAttributes.INPUT_VALUE, filepath)
            data = pd.read_csv(filepath)
            span.set_attribute("row_count", len(data))
            span.set_attribute("columns", str(data.columns.tolist()))
            span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(data.head(3).to_dict()))
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(data.head(3).to_dict()))
            return data
        except Exception as e:
            print(f"Warning: Couldn't load sample data from {filepath}: {e}")
            span.set_attribute("error", str(e))
            span.add_event("Error loading sample data", attributes={"error": str(e)})
            return None

def get_table_data(table_name):
    with tracer.start_as_current_span(
        "get_table_data",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.CHAIN.value}
    ) as span:
        """Get data from the specified table name"""
        if table_name.lower() in AVAILABLE_DATA_FILES:
            file_path = AVAILABLE_DATA_FILES[table_name.lower()]
            data = load_sample_data(file_path)
            span.set_attribute(SpanAttributes.RAW_INPUT, file_path)
            span.set_attribute(SpanAttributes.INPUT_VALUE, file_path)
            if data is not None:
                span.set_attribute("row_count", len(data))
                span.set_attribute("columns", str(data.columns.tolist()))
                span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(data.head(3).to_dict()))
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(data.head(3).to_dict()))
            span.set_attribute("file_path", file_path)
            return data, file_path
        else:
            span.set_attribute("error", f"Table name {table_name} not found in AVAILABLE_DATA_FILES")
            span.add_event("Table name not found", attributes={"table_name": table_name})
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
    with tracer.start_as_current_span(
        "set_formula_cell_value",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.TOOL.value}
    ) as span:
        # This is a simulation - in a real app, this would modify an actual spreadsheet
        print(f"\n[Tool Call] Setting formula in {sheet_name}!{cell_reference}:")
        print(f"==> {formula}")
        span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({
            "sheet_name": sheet_name,
            "cell_reference": cell_reference,
            "formula": formula
        }))
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({
            "sheet_name": sheet_name,
            "cell_reference": cell_reference,
            "formula": formula
        }))
        span.set_attribute("sheet_name", sheet_name)
        span.set_attribute("cell_reference", cell_reference)
        span.set_attribute("formula", formula)
        span.set_attribute(SpanAttributes.RAW_OUTPUT, f"Formula set in {sheet_name}!{cell_reference}")
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Formula set in {sheet_name}!{cell_reference}")
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
    with tracer.start_as_current_span(
        "execute_apps_script",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.TOOL.value}
    ) as span:
        # This is a simulation - in a real app, this would execute the script
        print(f"\n[Tool Call] {action_title}:")
        print(f"==> Running script: {code_string[:100]}{'...' if len(code_string) > 100 else ''}")
        span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({
            "code_string": code_string,
            "action_title": action_title
        }))
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({
            "code_string": code_string,
            "action_title": action_title
        }))
        span.set_attribute("code_string", code_string)
        span.set_attribute("action_title", action_title)
        span.set_attribute(SpanAttributes.RAW_OUTPUT, f"Executed: {action_title}")
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Executed: {action_title}")
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
    with tracer.start_as_current_span(
        "identify_table_name",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.CHAIN.value}
     ) as parent_span:
        if not client:
            return None, "ERROR: OpenAI client not initialized."

        available_tables = ", ".join(AVAILABLE_DATA_FILES.keys())
        parent_span.set_attribute("available_tables", available_tables)
        parent_span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({
            "user_query": user_query,
            "available_tables": available_tables
        }))
        parent_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({
            "user_query": user_query,
            "available_tables": available_tables
        }))
        system_prompt = (
            "You are an assistant that identifies which table or dataset a user is referring to in a query. "
            f"Available tables are: {available_tables}. "
            "ONLY return the exact table name you identified, or 'NONE' if no table is mentioned. "
            "Do not provide explanations or additional text."
        )
        
        try:
            with tracer.start_as_current_span("openai_call", 
                attributes={
                    SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.LLM.value
                }
            ) as llm_span:
                args = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Identify the table name in this query: '{user_query}'"}
                    ]
                }
                response = client.chat.completions.create(**args)

                table_name = response.choices[0].message.content.strip().lower()

                # Set LLM input/output attributes
                llm_span.set_attribute("llm.input_messages.0.message.role", "system")
                llm_span.set_attribute("llm.input_messages.0.message.content", system_prompt)
                llm_span.set_attribute("llm.input_messages.1.message.role", "user")
                llm_span.set_attribute("llm.input_messages.1.message.content", f"Identify the table name in this query: '{user_query}'")

                llm_span.set_attribute("llm.output_messages.0.message.role", "assistant")
                llm_span.set_attribute("llm.output_messages.0.message.content", response.choices[0].message.content)

                llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
                llm_span.set_attribute(SpanAttributes.LLM_PROVIDER, "openai")

                llm_span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(response.model_dump()))
                llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(response.model_dump()))
                llm_span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({k: v for k, v in args.items() if k != "client"}))
                llm_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({k: v for k, v in args.items() if k != "client"}))
                llm_span.set_attribute("table_name", table_name)

            # Check if the response is valid
            parent_span.set_attribute("table_name", table_name)
            if table_name == 'none':
                parent_span.set_attribute(SpanAttributes.RAW_OUTPUT, 'No table name was mentioned in your query. Please specify which table you want to analyze.')
                parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, 'No table name was mentioned in your query. Please specify which table you want to analyze.')
                parent_span.set_attribute("error", "No table name was mentioned in your query. Please specify which table you want to analyze.")
                return None, "ERROR: No table name was mentioned in your query. Please specify which table you want to analyze."
            
            if table_name in AVAILABLE_DATA_FILES:
                parent_span.set_attribute(SpanAttributes.RAW_OUTPUT, table_name)
                parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, table_name)
                return table_name, None
            else:
                parent_span.set_attribute(SpanAttributes.RAW_OUTPUT, f"ERROR: The identified table '{table_name}' is not available. Available tables are: {available_tables}")
                parent_span.add_event("Table name not found", attributes={"table_name": table_name})
                parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"ERROR: The identified table '{table_name}' is not available. Available tables are: {available_tables}")
                return None, f"ERROR: The identified table '{table_name}' is not available. Available tables are: {available_tables}"
                
        except Exception as e:
            parent_span.set_attribute(SpanAttributes.RAW_OUTPUT, f"ERROR: Could not identify table name: {e}")
            parent_span.add_event("Error identifying table name", attributes={"error": str(e)})
            parent_span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"ERROR: Could not identify table name: {e}")
            return None, f"ERROR: Could not identify table name: {e}"

def generate_excel_formula(user_query, table_data, table_name):
    """Generate an Excel formula using OpenAI based on the user query and specific table data"""
    with tracer.start_as_current_span(
        "generate_excel_formula",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.CHAIN.value}
    ) as span:
        import json
        span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({
            "user_query": user_query,
            "table_name": table_name
        }))
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({
            "user_query": user_query,
            "table_name": table_name
        }))
        span.set_attribute("user_query", user_query)
        span.set_attribute("table_name", table_name)
        
        if not client:
            error_msg = "ERROR: OpenAI client not initialized. Cannot generate formula."
            span.set_attribute(SpanAttributes.RAW_OUTPUT, error_msg)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
            span.add_event("Error generating formula", attributes={"error": error_msg})
            return error_msg

        if table_data is None:
            error_msg = "ERROR: No data available for the specified table."
            span.set_attribute(SpanAttributes.RAW_OUTPUT, error_msg)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
            span.add_event("Error generating formula", attributes={"error": error_msg})
            return error_msg
        
        # Prepare data context information
        columns = ", ".join(table_data.columns.tolist())
        row_count = len(table_data)
        # Convert DataFrame to string representation for tracing
        sample_data_str = str(table_data.head(3).to_dict())
        
        span.set_attribute("columns", columns)
        span.set_attribute("row_count", row_count)
        span.set_attribute("sample_data", sample_data_str[:1000])  
        
        data_context = f"""
Table name: {table_name}
File: {AVAILABLE_DATA_FILES.get(table_name, 'Unknown')}
Columns: {columns}
Row count: {row_count}
Sample data (first 3 rows): {table_data.head(3).to_dict()}
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
            with tracer.start_as_current_span("openai_call", 
                attributes={
                    SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.LLM.value
                }
            ) as llm_span:
                args = {
                    "model": "gpt-4o", 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 500
                }
                
                response = client.chat.completions.create(**args)
                formula = response.choices[0].message.content.strip()
                
                llm_span.set_attribute("llm.input_messages.0.message.role", "system")
                llm_span.set_attribute("llm.input_messages.0.message.content", system_prompt)
                llm_span.set_attribute("llm.input_messages.1.message.role", "user")
                llm_span.set_attribute("llm.input_messages.1.message.content", user_prompt)
                
                llm_span.set_attribute("llm.output_messages.0.message.role", "assistant")
                llm_span.set_attribute("llm.output_messages.0.message.content", formula)
                
                llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4o")
                llm_span.set_attribute(SpanAttributes.LLM_PROVIDER, "openai")
                
                # Fix: Convert response and args to JSON strings for raw input/output
                import json
                llm_span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(response.model_dump()))
                llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(response.model_dump()))
                llm_span.set_attribute(SpanAttributes.RAW_INPUT, json.dumps({k: v for k, v in args.items() if k != "client"}))
                llm_span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps({k: v for k, v in args.items() if k != "client"}))
                llm_span.set_attribute("formula", formula)
            
            # Simple validation - ensure it looks like a formula
            if not formula.startswith('='):
                formula = f"={formula}"
            
            span.set_attribute("formula", formula)
            span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(formula))
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(formula))
            return formula
        except Exception as e:
            error_msg = f"ERROR: Could not generate formula: {e}"
            span.set_attribute(SpanAttributes.RAW_OUTPUT, error_msg)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
            span.add_event("Error generating formula", attributes={"error": error_msg})
            return error_msg

# --- Agent Logic ---

def process_query(user_query):
    """Process the user query through the complete workflow to return an Excel formula"""
    with tracer.start_as_current_span(
        "process_query",
        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.AGENT.value}
    ) as span:
        span.set_attribute("user_query", user_query)
        span.set_attribute(SpanAttributes.INPUT_VALUE, user_query)
        span.set_attribute(SpanAttributes.RAW_INPUT, user_query)
        # Step 1: Identify which table the user is referring to
        print("Step 1: Identifying table name mentioned in query...")
        table_name, error = identify_table_name(user_query)
        
        if error:
            print(f"Error identifying table: {error}")
            span.set_attribute(SpanAttributes.RAW_OUTPUT, error)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, error)
            span.add_event("Error identifying table", attributes={"error": error})
            return error
        
        print(f"Identified table: {table_name}")
        span.set_attribute("table_name", table_name)
        
        # Step 2: Load the specific table data
        print(f"Step 2: Loading data for table '{table_name}'...")
        table_data, file_path = get_table_data(table_name)
        
        if table_data is None:
            error_msg = f"ERROR: Could not load data for table '{table_name}'"
            print(error_msg)
            span.set_attribute(SpanAttributes.RAW_OUTPUT, error_msg)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, error_msg)
            span.add_event("Error loading data", attributes={"error": error_msg})
            return error_msg
        
        print(f"Loaded {len(table_data)} rows from {file_path}")
        span.set_attribute("row_count", len(table_data))
        span.set_attribute("file_path", file_path)
        
        # Step 3: Generate the Excel formula
        print("Step 3: Generating Excel formula...")
        formula = generate_excel_formula(user_query, table_data, table_name)
        span.set_attribute("formula", formula)
        
        # Step 4: Simulate setting the formula (if valid)
        if not formula.startswith("ERROR:"):
            set_formula_cell_value("Results", "A1", formula)
        
        span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(formula))
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(formula))
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
    # with tracer.start_as_current_span(
    #     "run_hardcoded_mode",
    #     attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.AGENT.value}
    # ) as span:
        print("\n--- Excel Formula Generator (Hardcoded Mode) ---")
        print("Running predetermined examples:")
        
        results = []
        for i, example_query in enumerate(HARDCODED_EXAMPLES, 1):
            print(f"\nExample {i}: {example_query}")
            formula = process_query(example_query)
            results.append({"query": example_query, "formula": formula})
            
            if formula.startswith("ERROR:"):
                print(f"❌ {formula}")
            else:
                print(f"✅ Formula: {formula}")
        
        print("\nHardcoded examples completed.")


def run_interactive_mode():
        """Run the interactive mode."""
    # with tracer.start_as_current_span(
    #     "run_interactive_mode",
    #     attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.AGENT.value}
    # ) as span:
        
        print("\n--- Excel Formula Generator (Interactive Mode) ---")
        print("Enter your data analysis query (or type 'quit' to exit).")
        print(f"Available tables: {', '.join(AVAILABLE_DATA_FILES.keys())}")

        session_queries = []
        try:
            while True:
                try:
                    user_input = input("\n> ")
                    if user_input.lower() in ('quit', 'exit', 'q'):
                        break
                    if not user_input.strip():
                        continue
                    
                    with tracer.start_as_current_span(
                        "process_user_input",
                         attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.CHAIN.value}
                    ) as input_span:
                        input_span.set_attribute(SpanAttributes.RAW_INPUT, user_input)
                        input_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)
                        input_span.set_attribute("user_input", user_input)
                        formula = process_query(user_input)
                        
                        if formula.startswith("ERROR:"):
                            print(f"\n❌ {formula}")
                            input_span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(formula))
                            input_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(formula))
                        else:
                            print(f"\n✅ Generated Formula: {formula}")
                            input_span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(formula))
                            input_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(formula))
                        session_queries.append({"query": user_input, "formula": formula})
                        input_span.set_attribute(SpanAttributes.RAW_OUTPUT, json.dumps(formula))
                        input_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(formula))
                        input_span.set_attribute("formula", formula)
                    
                except EOFError:
                    # Handle Ctrl+D
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    break
                except Exception as e:
                    error_msg = f"Error processing input: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    with tracer.start_as_current_span(
                        "process_error",
                        attributes={SpanAttributes.FI_SPAN_KIND: FiSpanKindValues.UNKNOWN.value}
                    ) as error_span:
                        error_span.set_attribute("error", error_msg)
                        error_span.set_attribute("user_input", user_input)
            
            print("\nExiting Excel Formula Generator.")
        
        except Exception as e:
            raise

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