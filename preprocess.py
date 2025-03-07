import os
import pandas as pd
import re
from Config import Config


def get_input_data() -> pd.DataFrame:
    """
    Loads CSV data from the data folder, renames columns according to configuration,
    concatenates datasets, and returns a single DataFrame with a unified label column "y".

    Returns:
        pd.DataFrame: Combined data from AppGallery.csv and Purchasing.csv with unified label column.
    """
    try:
        data_folder = os.path.join(os.getcwd(), "data")
        appgallery_path = os.path.join(data_folder, "AppGallery.csv")
        purchasing_path = os.path.join(data_folder, "Purchasing.csv")

        df1 = pd.read_csv(appgallery_path, skipinitialspace=True)
        df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
        df2 = pd.read_csv(purchasing_path, skipinitialspace=True)
        df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

        df = pd.concat([df1, df2], ignore_index=True)

        # Ensure text columns are strings.
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

        # Filter out rows where y2 is empty or missing.
        df = df.loc[(df["y2"] != "") & (~df["y2"].isna()),]

        # Create a unified label column "y" using the column defined in Config.CLASS_COL.
        df["y"] = df[Config.CLASS_COL]
        return df

    except Exception as e:
        print(f"Error loading input data: {e}")
        raise


def de_duplication(data: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate and process the interaction content for each ticket.

    Args:
        data (pd.DataFrame): Input data with raw interaction content.

    Returns:
        pd.DataFrame: DataFrame with deduplicated and concatenated interaction content.
    """
    data["ic_deduplicated"] = ""

    # Define customer support templates (for English).
    cu_template = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"
        ]
    }
    cu_pattern = ""
    for pattern in sum(list(cu_template.values()), []):
        cu_pattern += f"({pattern})|"
    cu_pattern = cu_pattern[:-1]

    # Email split patterns.
    pattern_1 = r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = r"(On.{30,60}wrote:)"
    pattern_3 = r"(Re\s?:|RE\s?:)"
    pattern_4 = r"(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = r"(\s?\*\*\*\*\*\(PHONE\))*$"
    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        df_ticket = data.loc[data['Ticket id'] == t,]
        ic_set = set()
        ic_deduplicated = []
        for ic in df_ticket[Config.INTERACTION_CONTENT]:
            ic_parts = re.split(split_pattern, ic)
            ic_parts = [i for i in ic_parts if i is not None]
            ic_parts = [re.sub(split_pattern, "", i.strip()) for i in ic_parts]
            ic_current = []
            for part in ic_parts:
                if part and part not in ic_set:
                    ic_set.add(part)
                    ic_current.append(part + "\n")
            ic_deduplicated.append(' '.join(ic_current))
        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated

    # Save intermediate output to out.csv.
    output_path = os.path.join(os.getcwd(), "out.csv")
    data.to_csv(output_path, index=False)

    # Replace original interaction content with deduplicated content.
    data[Config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data.drop(columns=['ic_deduplicated'], inplace=True)
    return data


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove noise patterns from the Ticket Summary and Interaction Content.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(noise, " ", regex=True).replace(r'\s+',
                                                                                                              ' ',
                                                                                                              regex=True).str.strip()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()

    good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    df = df.loc[df.y1.isin(good_y1)]
    return df
