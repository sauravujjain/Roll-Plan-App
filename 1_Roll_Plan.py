import streamlit as st
import pandas as pd
import numpy as np
import random
from io import BytesIO



# Example of an Excel file
#file_path = 'path/to/your/template.xlsx'  # Update this to the path of your Excel file

# Create a file uploader widget
#uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Check if a file has been uploaded
#if uploaded_file is not None:
    # Use Pandas to read the Excel file for each sheet
    #with pd.ExcelFile(uploaded_file) as xls:
        # Assuming the names of the sheets are 'cutplan' and 'rolls_data'
        #df_cutplan = pd.read_excel(xls, 'cutplan')
       #df_rolls_data = pd.read_excel(xls, 'rolls_data')

# Title of the app
st.title("Roll Planning App")
# Function to initialize or access session state variables
def init_state():
    if 'step' not in st.session_state:
        st.session_state.step = 0

# Increment the step in the workflow
def next_step():
    st.session_state.step += 1

# Decrement the step (if you want to allow going back)
def previous_step():
    if st.session_state.step > 0:
        st.session_state.step -= 1

def jump_to_ra():
    st.session_state.step = 3

def restart():
    st.session_state.step = 0

# Initialize session state
init_state()


if st.session_state.step == 0:

    st.subheader("Upload Cutplan And Fabric Details")
    # Provide a button to download the sample file
    # Direct reference to the file name in the same directory
    with open("Cutplan Data Template.xlsx", "rb") as file:
        st.download_button(
            label="Download Sample Excel File",
            data=file,
            file_name="Cutplan Data Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Upload Cutplan Excel file", type=["xlsx"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Use Pandas to read the Excel file for each sheet
        with pd.ExcelFile(uploaded_file) as xls:
            # Assuming the names of the sheets are 'cutplan' and 'rolls_data'
            cutplan_df = pd.read_excel(xls, 'cutplan')
            rolls_df = pd.read_excel(xls, 'rolls_data')
        
        st.session_state.cutplan_df = cutplan_df
        st.session_state.rolls_df = rolls_df
        st.dataframe(cutplan_df)
        st.dataframe(rolls_df)

        st.write("Press to continue to Roll Plan")
        st.button("Go to Roll Plan", on_click = jump_to_ra)

    if uploaded_file is None:
        st.subheader("If Upload data is not available click here to create Demo data" )
        st.button("Next", on_click = next_step)

        

    
    

if st.session_state.step == 1:

    st.subheader("Create Fabric Rolls")
    avg_roll_size = st.number_input("Enter the Average Roll Length:", min_value=0.0, value=100.0, step=0.1)
    fab_quantity = st.number_input("Input the total Fabric quantity:", min_value=0.0, value=1000.0, step=0.1)

    with st.expander("More Settings"):
        Length_variation_range = st.number_input("Enter value for the roll legth variation range", min_value = 0.0, value =30.0, step=0.1)
    if st.button('Create Roll data'):
        # Place the logic that needs to be executed when the button is pressed
        iterations = 0
        rolls_list = []  
        while not (-1 < (fab_quantity - sum(rolls_list)) < 1):
            if iterations >= 1000:
                print("Maximum iterations reached, exiting loop.")
                break
            rolls_list = []    
            for roll in range(0, int(fab_quantity//avg_roll_size)):
                roll_length = round(random.uniform((avg_roll_size - Length_variation_range) ,(avg_roll_size + Length_variation_range)), 2)
                rolls_list.append(roll_length)  
            
            iterations += 1
        st.write(f"Rolls_List: {rolls_list}")
        st.write(f"Sum of Rolls: {round(sum(rolls_list), 2)}")
        st.write(f"Difference from required sum of Rolls : {round((fab_quantity - sum(rolls_list)),2)}")
        # Display DataFrame using Streamlit
        rolls_list_data = {"Roll_Number" :[f"R{i}" for i in range (1, len(rolls_list)+1)] , "Roll_Length" : rolls_list}
        rolls_df = pd.DataFrame(rolls_list_data)
        st.session_state.rolls_df = rolls_df
        st.dataframe(rolls_df)
        #st.download_button(("Download Roll List"), rolls_list_df.to_csv(index=False), file_name="rolls_data.csv", mime = 'test/csv')


    st.write("Press Next to continue to Cutplan")
    st.button("Next", on_click = next_step)


elif st.session_state.step == 2:
    with st.expander("See Rolls Data"):
        st.dataframe(st.session_state.rolls_df)

    st.subheader("Create Sample Cutplan")
    
    max_marker_length = st.number_input("Enter the Max Marker Length:", min_value=0.0, value=5.0, step=0.1)
    number_of_markers = st.number_input("Enter the number of markers required:", min_value=1, value=5, step=1)
    buffer_percentage = st.number_input("Provide extra fabric buffer in integer percentages", min_value=0, value=5, step=1)

    st.write("Step1: Click to Create Demo Markers")                                   
    if st.button('Create Demo Markers'):

        marker_list = sorted([round(random.uniform(1, max_marker_length) , 2) for _ in range (0,number_of_markers)],reverse = True)

        marker_length_data = {"Marker_Number" :[i for i in range (1, len(marker_list)+1)] , "Marker_Length" : marker_list}
        marker_df = pd.DataFrame(marker_length_data)
        st.session_state.marker_df = marker_df
        st.dataframe(marker_df)
        
        #st.download_button(("Download Marker Length data"), marker_data_df .to_csv(index=False), file_name="marker_data.csv", mime = 'test/csv')
    st.write("Step2: Click to Create Demo Cutplan")
    if st.button('Create Demo Cutplan'):
        rolls_list = st.session_state.rolls_df["Roll_Length"]
        buffer = round((sum(rolls_list)*buffer_percentage)/100 , 1)
        numbers = st.session_state.marker_df["Marker_Length"]
        target = np.float32(sum(rolls_list)- buffer)
        max_factor = min(int(sum(rolls_list)/2),500)
        population_size = 100
        num_generations = 1000
        mutation_rate = 0.01
        
        def initialize_population():
            return [np.random.randint(1, max_factor + 1, len(numbers)) for _ in range(population_size)]
        
        def fitness(individual):
            individual = sorted(individual, reverse=True)
            products = [a * b for a, b in zip(individual, numbers)]
            diff_from_target = abs(target - sum(products))
            fitness_score = diff_from_target 
            return (fitness_score)  # Lower is better

        def select_parents(population):
            fitnesses = np.array([fitness(individual) for individual in population])
            probabilities = 1 / (1 + fitnesses)  # lower fitness -> higher probability
            probabilities /= probabilities.sum()
            parents_indices = np.random.choice(range(population_size), size=population_size//2, p=probabilities)
            return [population[i] for i in parents_indices]

        def crossover(parent1, parent2):
            crossover_point = np.random.randint(1, len(numbers))
            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return offspring

        def mutate(individual):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(len(numbers))
                individual[mutation_point] = np.random.randint(1, max_factor + 1)
            return individual
        
        # Genetic Algorithm
        population = initialize_population()
        for _ in range(num_generations):
            new_population = []
            parents = select_parents(population)
            for i in range(len(parents)):
                parent1, parent2 = parents[i], parents[(i+1)% len(parents)]
                offspring1, offspring2 = crossover(parent1, parent2), crossover(parent2, parent1)
                new_population.extend([mutate(offspring1), mutate(offspring2)])
                
            population = new_population
            
        best_solution = sorted(min(population, key=fitness), reverse=True)

        cutplan_list = {"Marker_Name" : [f"M{i}" for i in range(1,len(numbers)+1)], "Marker_Length" : numbers, "Ply_Height" : best_solution , "Required_Fabric" : numbers*best_solution}
        cutplan_df = pd.DataFrame(cutplan_list)
        st.dataframe(cutplan_df)
        st.session_state.cutplan_df = cutplan_df
        st.write("Total Target Fabric: ", target)
        st.write("Total planned Fabric: ", np.sum(np.array(numbers)*best_solution))
        st.write("Difference:",target-(np.sum(np.array(numbers)*best_solution)))
        

        # Specify the filename
        def generate_excel(df1, df2):
            output = BytesIO()
            filename = 'Cutplan.xlsx'   
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write each DataFrame to a specific sheet
                df1.to_excel(writer, sheet_name='cutplan', index=False)
                df2.to_excel(writer, sheet_name='rolls_data', index=False)

                # Auto-adjust columns' width
                for sheetname, df in {'cutplan': df1, 'rolls_data': df2}.items():
                    worksheet = writer.sheets[sheetname]
                    for idx, col in enumerate(df):  # loop through all columns
                        max_len = max(df[col].astype(str).apply(len).max(),  # length of largest item
                                len(col))  # length of column name/header
                        worksheet.set_column(idx, idx, max_len+1)  # set column width
                
            output.seek(0)
            return output

        excel_file = generate_excel(st.session_state.cutplan_df, st.session_state.rolls_df)

        st.download_button(
        label="Download Cutplan and Fabric Excel",
        data=excel_file,
        file_name="Cutplan.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.write("Press Next to continue to RollPlan")
        st.button("Next", on_click = next_step)
    
        


elif st.session_state.step == 3:

    with st.expander("See Cutplan and Rolls Data"):
        st.dataframe(st.session_state.rolls_df)
        st.dataframe(st.session_state.cutplan_df)


    if st.button('Create Roll Plan'):
        population_size = 200
        num_generations = 500
        mutation_rate = 0.05
        
        def initialize_population_1(max_attempts=1000):
            population = []
            for _ in range(population_size):
                for attempt in range(max_attempts):
                    if len(rolls)>1:
                        individual_size = np.random.randint(1,len(rolls)+1)
                    else:
                        individual_size = len(rolls)
                    individual_indices = np.random.choice(len(rolls), size=individual_size, replace=False)
                    individual = [rolls[i] for i in individual_indices]
                    individual_lengths = [length for _, length in individual]  # Extract lengths from tuple
                    sum_of_remainders = sum(length % marker_length for length in individual_lengths)
                    # Check if the sum is within the target range
                    if all(length > marker_length for length in individual_lengths):
                        if target <= sum(individual_lengths)- sum_of_remainders: #<= target + (2*tolerance):
                            population.append(individual)
                            break
            return population
            
        def initialize_population_2(max_attempts=1000):
            population = []
            for _ in range(population_size):
                for attempt in range(max_attempts):
                    if len(rolls)>1:
                        individual_size = np.random.randint(1,len(rolls)+1)
                    else:
                        individual_size = len(rolls)     
                    individual_indices = np.random.choice(len(rolls), size=individual_size, replace=False)
                    individual = [rolls[i] for i in individual_indices]
                    individual_lengths = [length for _, length in individual]  # Extract lengths from tuple
                    #sum_of_remainders = sum(length % marker_length for length in individual_lengths)
                    # Check if the sum is within the target range
                    if all(length > marker_length for length in individual_lengths):
                        population.append(individual)
                        break
            return population

        def fitness_1(individual):
    
            individual_lengths = [length for _, length in individual]  # Extract lengths
            # Component 1: Difference between sum of rolls and target
            diff_from_target = sum(individual_lengths) - target

            # Component 2: Sum of remainders divided by marker_length
            sum_of_remainders = sum(length % marker_length for length in individual_lengths)
                            
            if sum(individual_lengths)-sum_of_remainders < target:
                fitness_score = 1e100
                
            else : 
                fitness_score = diff_from_target

            return fitness_score #Lower is better

        def fitness_2(individual):
    
            individual_lengths = [length for _, length in individual]  # Extract lengths
            # Component 1: Sum of remainders divided by marker_length
            sum_of_remainders = sum(length % marker_length for length in individual_lengths)
            # Component 1: Difference between sum of rolls and target
            diff_from_target = abs(sum(individual_lengths) - target - sum_of_remainders)
            fitness_score = diff_from_target

            return fitness_score #Lower is better  

        def tournament_selection(population, tournament_size=5):
            #if len(population) == 1:
            # return population
            
            selected_parents = []
            for _ in range(len(population) // 2):
                tournament_indices = np.random.choice(len(population), tournament_size, replace=True)
                tournament = [population[i] for i in tournament_indices]
                best_individual = min(tournament, key=fitness_1)
                selected_parents.append(best_individual)
            return selected_parents

        def tournament_selection_2(population, tournament_size=5):
            #if len(population) == 1:
            # return population
            
            selected_parents = []
            for _ in range(len(population) // 2):
                tournament_indices = np.random.choice(len(population), tournament_size, replace=True)
                tournament = [population[i] for i in tournament_indices]
                best_individual = min(tournament, key=fitness_2)
                selected_parents.append(best_individual)
            return selected_parents

        def crossover_1(parent1, parent2):
            # Combine unique rolls from both parents
            unique_rolls = list(set(parent1 + parent2))
            
            # Ensure the crossover point is within the smaller range of unique rolls or parents
            min_length = min(len(unique_rolls), len(parent1), len(parent2))
            
            # Handle cases where a meaningful crossover cannot occur
            #if min_length <= 1:
                # Option 1: Return the parents as is
                #return parent1, parent2
                # Option 2: Implement other logic suitable for your scenario
            
            #else:
                # Adjusted to ensure mixing, ensuring there's at least one element for crossover
            crossover_point = random.randint(0, min_length)
            
            # Initialize offspring with unique rolls up to the crossover point
            offspring1 = unique_rolls[:crossover_point]
            offspring2 = unique_rolls[crossover_point:]
            
            # Fill the rest of the offspring with rolls from the other parent, avoiding duplicates
            additional_rolls1 = [roll for roll in parent2 if roll not in offspring1]
            additional_rolls2 = [roll for roll in parent1 if roll not in offspring2]
            offspring1.extend(additional_rolls1[:len(parent1) - len(offspring1)])
            offspring2.extend(additional_rolls2[:len(parent2) - len(offspring2)])
            
            return offspring1, offspring2


        def mutate_1(individual):
            if not individual:
                return individual  # Return the individual unchanged if it's empty

            if random.random() < mutation_rate:
                i = random.randint(0,len(individual)-1)
                available_rolls = [(name, length) for name, length in rolls if (name, length) not in individual and length > marker_length]
                if available_rolls:
                    individual[i] = random.choice(available_rolls)   

            return individual


    # Assuming cutplan is a DataFrame with columns 'Marker_Length' and 'Ply_Height'
        total_residual = 0
        reused_residuals =set()
        rolls = list(zip(st.session_state.rolls_df['Roll_Number'], st.session_state.rolls_df['Roll_Length']))

        # Initialize an empty list to collect data
        ra_reports = []

        for marker_name, marker_length, ply_height in zip(st.session_state.cutplan_df["Marker_Name"], st.session_state.cutplan_df["Marker_Length"], st.session_state.cutplan_df["Ply_Height"]):
            
            target = marker_length * ply_height
            population = initialize_population_1()  # Make sure this function is correctly defined

            if not population: # Check if the population is empty
                # If empty, call the alternate initialization function
                population = initialize_population_2()
                for _ in range(num_generations):
                    new_population = []
                    parents = tournament_selection_2(population)  # Ensure this function is defined

                    for i in range(0, len(parents)):
                        parent1, parent2 = parents[i], parents[(i+1) % len(parents)]
                        offspring1, offspring2 = crossover_1(parent1, parent2)  # Check this function
                        new_population.extend([mutate_1(offspring1), mutate_1(offspring2)])  # And this one
                        #new_population.extend([offspring1, offspring2])  # removing mutation function
                    population = new_population

                best_solution = sorted(min(population, key=lambda x: fitness_2(x)), key = lambda X: X[1] %marker_length) # Define fitness function

            for _ in range(num_generations): # Main default GA Loop
                new_population = []
                parents = tournament_selection(population)  # Ensure this function is defined

                for i in range(0, len(parents)):
                    parent1, parent2 = parents[i], parents[(i+1) % len(parents)]
                    offspring1, offspring2 = crossover_1(parent1, parent2)  # Check this function
                    new_population.extend([mutate_1(offspring1), mutate_1(offspring2)])  # And this one
                    
                population = new_population

            best_solution = sorted(min(population, key=lambda x: fitness_1(x)), key = lambda X: X[1] %marker_length) # Define fitness functio
            max_plies_possible = 0
            plies_planned = 0
            residuals=[]
            ply_per_roll=[]
            for roll in best_solution:
                max_plies_possible += roll[1]//marker_length
                if max_plies_possible <= ply_height:
                    residuals.append((f"{roll[0]}-bit" , round(roll[1]%marker_length,2)))
                    plies_planned += roll[1]//marker_length
                    ply_per_roll.append(roll[1]//marker_length)
                else:
                    residuals.append((f"{roll[0]}-bit" , round(roll[1] - marker_length*(ply_height-plies_planned),2)))
                    ply_per_roll.append(ply_height-plies_planned)
            
            if sum(ply_per_roll) - ply_height == 0:
                roll_plan_status = "Successful"
            else:
                roll_plan_status = "Partial Roll PLan"

            st.write("Marker Name:", marker_name, "Marker Length:", marker_length , "Required_Fabric:", round(target,2) , "Roll Plan Status:", roll_plan_status )
            st.write("*******************************************************************")
            
            
            # Create DataFrame or dictionary for reporting
            ra_report = {
                "Marker Name": marker_name,
                "Marker Length": marker_length,
                "Plies Required": ply_height,
                "Plies Planned": sum(ply_per_roll),
                "Fabric Required": round(target, 2),
                "Fabric Selected": round(sum(length for _,length in best_solution), 2),
                "Planned Rolls": best_solution,  # storing only the IDs or descriptions
                "End Bits": residuals,  # storing only the descriptions
                "End Bits Sum": round(sum([length for _,length in residuals]), 2)
            }

           
            ra_reports.append(ra_report)
            total_residual += sum([length for _,length in residuals])

            for roll in best_solution:
                if roll in rolls:
                    rolls.remove(roll)
                    
            # Add residuals that meet the criterion back to the rolls list       
            for residual in residuals:
                if residual[1] > min(st.session_state.cutplan_df["Marker_Length"]) and residual not in reused_residuals:
                    rolls.append(residual)
                    reused_residuals.add(residual)

        total_residual -= sum([length for _,length in reused_residuals])  

        ra_df = pd.DataFrame(ra_reports)

        ra_df["Planned Rolls"] = ra_df["Planned Rolls"].astype(str)
        ra_df["End Bits"] = ra_df["End Bits"].astype(str)
        
        st.dataframe(ra_df)
        st.session_state.ra_df = ra_df

        st.write("*******************Solution Complete*****************")
        st.write("Total residual",  round(total_residual,2))
        st.write("Residual % = " , round(total_residual/sum(st.session_state.rolls_df["Roll_Length"]),2))


        st.subheader("Press Restart to start afresh")
        st.button("Restart", on_click = restart)


