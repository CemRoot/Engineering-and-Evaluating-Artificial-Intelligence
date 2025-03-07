import pandas as pd
import re
from Config import *


def load_input_data() -> pd.DataFrame:
    """
    Loads input data from CSV files.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        # Load data from CSV files
        df1 = pd.read_csv("data/AppGallery.csv", skipinitialspace=True)
        df2 = pd.read_csv("data/Purchasing.csv", skipinitialspace=True)
        
        # Rename columns for consistency
        df1 = df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'})
        df2 = df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'})
        
        # Concatenate dataframes
        df = pd.concat([df1, df2])
        
        # Ensure data types for interaction content and ticket summary
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype('U')
        
        # Filter rows with empty or null values in column 'y'
        df["y"] = df[Config.CLASS_COL]
        df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
        
        return df
    
    except Exception as e:
        print(f"Error loading input data: {e}")
        return None


def deduplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicates from interaction content.
    
    Args:
        df (pd.DataFrame): Input data.
    
    Returns:
        pd.DataFrame: Data with duplicates removed.
    """
    try:
        # Customer template patterns
        cu_template = {
            # ...
        }
        
        # Email split patterns
        pattern_1 = "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
        pattern_2 = "(On.{30,60}wrote:)"
        pattern_3 = "(Re\s?:|RE\s?:)"
        pattern_4 = "(\*\*\*\*\*\(PERSON\) Support issue submit)"
        pattern_5 = "(\s?\*\*\*\*\*\(PHONE\))*$"
        
        split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"
        
        # Process ticket data
        tickets = df["Ticket id"].value_counts()
        
        for t in tickets.index:
            # Process interaction content for each ticket
            df_ticket = df.loc[df['Ticket id'] == t,]
            ic_set = set([])
            ic_deduplicated = []
            
            for ic in df_ticket[Config.INTERACTION_CONTENT]:
                # Split and clean interaction content
                ic_r = re.split(split_pattern, ic)
                ic_r = [i for i in ic_r if i is not None]
                ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]
                ic_r = [re.sub('|'.join(cu_template.values()), "", i.strip()) for i in ic_r]
                
                # Add unique content
                ic_current = []
                for i in ic_r:
                    if len(i) > 0 and i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current.append(i)
                
                ic_deduplicated.extend([' '.join(ic_current)])
            
            df.loc[df["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated
        
        # Save processed data to CSV file
        df.to_csv('out.csv', index=False)
        
        # Replace interaction content with deduplicated version
        df[Config.INTERACTION_CONTENT] = df['ic_deduplicated']
        df = df.drop(columns=['ic_deduplicated'])
        
        return df
    
    except Exception as e:
        print(f"Error deduplicating interactions: {e}")
        return None

Key Differences:
Function Name Change:

Original: get_input_data
Updated: load_input_data
Error Handling:

The updated code includes a try...except block to handle exceptions and print an error message.
CSV File Paths:

The original code uses double slashes (//), while the updated code uses single slashes (/).
Return Type:

Both functions return a pd.DataFrame, but the updated function explicitly handles potential errors by returning None.
Comments and Documentation:

The updated function includes a docstring to describe its purpose and return value.
