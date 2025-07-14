# Roll Planning Optimization App

![App Screenshot](https://via.placeholder.com/800x400?text=Roll+Planning+App+Screenshot)

This Streamlit application optimizes fabric roll allocation for garment production, helping manufacturers minimize waste and improve efficiency. The app guides users through a step-by-step workflow to create cut plans, generate roll data, and optimize fabric usage.

## Key Features

- **Multi-Step Workflow**: Guided process from data input to optimization
- **Flexible Data Input**: Upload existing Excel files or generate sample data
- **Intelligent Roll Generation**: Create realistic fabric roll datasets
- **Cutplan Optimization**: Generate optimized synthetic cut plans for testing, which mathes the created fabric using genetic algorithms
- **Residual Management**: Track and reuse fabric remnants across production
- **Comprehensive Reporting**: Detailed roll allocation reports with waste metrics

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Numpy

## Installation

```bash
pip install streamlit pandas numpy
```

## Workflow

### Step 1: Upload Data or Generate Demo
- Download Excel template for cutplan and roll data
- Upload existing Excel files OR
- Generate demo fabric roll data

### Step 2: Create Fabric Rolls
- Specify average roll length and total fabric quantity
- Set roll length variation parameters
- Generate randomized roll dataset

### Step 3: Create Cutplan
- Define marker parameters (max length, quantity)
- Set fabric buffer percentage
- Generate optimized cutplan using genetic algorithms
- Download complete cutplan Excel file

### Step 4: Run Roll Plan Optimization
- Execute genetic algorithm to optimize roll allocation
- View detailed roll assignment reports
- Analyze fabric utilization and residual waste
- Restart workflow for new scenarios

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Follow the step-by-step workflow:
   - Upload Excel files OR generate demo data
   - Configure and generate fabric rolls
   - Create optimized cutplan
   - Run roll allocation optimization

3. Analyze results:
   - Marker-specific roll assignments
   - Fabric utilization metrics
   - Residual waste percentages
   - Production efficiency statistics

## Technical Highlights

- **Genetic Algorithm Optimization**: Efficient roll allocation using evolutionary computation
- **Residual Reuse System**: Automatically recycles usable fabric remnants
- **Dynamic Workflow**: State management for seamless step transitions
- **Excel Integration**: Import/export capabilities for production data
- **Statistical Roll Generation**: Realistic fabric roll modeling

## File Templates

### Cutplan Excel Structure
- **cutplan** sheet:
  - Marker_Name
  - Marker_Length
  - Ply_Height
  - Required_Fabric

- **rolls_data** sheet:
  - Roll_Number
  - Roll_Length

## License
MIT License
