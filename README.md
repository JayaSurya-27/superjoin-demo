# Excel Formula Generator Agent

This project provides an agent that converts natural language queries into Excel formulas. It can analyze data from sample tables and generate appropriate Excel formulas to perform the requested operations.

## Prerequisites

- Python 3.9+ 
- OpenAI API key
- FI_API_KEY
- FI_SECRET_KEY

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   FI_API_KEY=your-api-key-here
   FI_SECRET_KEY=your-secret-key-here
   ```

## Usage

### Running Hardcoded Examples

To run the agent with predefined example queries:

```bash
python excel_agent.py
```

### Interactive Mode

To run the agent in interactive mode where you can enter your own queries:

```bash
python excel_agent.py --interactive
```

In interactive mode, you can:
- Type your natural language queries about the available data tables
- Enter 'quit', 'exit', or 'q' to exit the program

### Available Data Tables

The following sample data tables are available:
- `sales_q1` or `q1`: Sales data for Q1
- `sales_q2` or `q2`: Sales data for Q2
- `customers` or `customer`: Customer data

Example queries:
- "Calculate the total sales amount in the sales_q1 table"
- "Find the average order value across all records in sales_q2"
- "Count how many unique customers we have in the customers table"

## Project Structure

- `excel_agent.py`: Main script containing the agent logic
- `excel_agent_eval_tags.py`: Evaluation tags for telemetry
- Sample data files (CSV format)