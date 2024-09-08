from typing import Tuple, Literal, Optional, Type

import pandas
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from pandas import DataFrame
from pydantic import BaseModel

from experiments.old.dataclass_ import LIST_COMMON_COLUMNS

UNNAME_PREFIX = "Unnamed: "


def get_skip_row(dataframe: pandas.DataFrame) -> int:
    """
    Get the number of rows to skip

    if the column contains 'ean', this is
    the first row, skip_row is the next row
    """
    for index_find in range(99):
        try:
            list_columns = [str(column).upper() for column in dataframe.iloc[index_find]]
        except IndexError as exception:
            break

        for common_column in LIST_COMMON_COLUMNS:
            if common_column in list_columns:
                return index_find


def strip_unnamed_column(
    dataframe: pandas.DataFrame,
    increment: Literal[1, -1],
    skip: int = 0,
) -> Tuple[pandas.DataFrame, int, int]:
    """
    Strip 'unname' columns from the dataframe

    browse the first and last columns, if theses columns
    contains "Noname", remove them and recursively call the function

    Args:
        dataframe (pandas.DataFrame): the dataframe to process
        increment (int): the increment to use to browse the columns
        skip (int): the number of columns to skip

    Returns:
        Tuple[pandas.DataFrame, int, int]: the dataframe without the 'unname' columns, increment, skip value
    """
    # Start at the first or last column (depending on increment)
    index = 0 if increment == 1 else -1
    column = dataframe.columns[index]

    # Column can be a int, float...
    condition = (
        isinstance(column, int),
        isinstance(column, float),
        UNNAME_PREFIX in f"{column}",
    )

    # Drop the column, recursively call the function
    if any(condition):
        dataframe = dataframe.drop(columns=column)

        return strip_unnamed_column(
            dataframe=dataframe,
            increment=increment,
            skip=skip + increment,
        )

    return dataframe, increment, skip


def split_noname_column(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    """
    Split the dataframe to remove the column with 'Noname'

    Args:
        dataframe (pandas.DataFrame): the dataframe to process

    Returns:
        pandas.DataFrame: the dataframe first split at the column with 'Noname'
    """
    generator = (index for index, column in enumerate(dataframe.columns) if UNNAME_PREFIX in f"{column}")
    result = next(generator, None)

    if result is not None:
        return dataframe.iloc[:, :result]

    return dataframe


def drop_rows_after_nan(dataframe: DataFrame) -> DataFrame:
    """
    Drops all rows after and including the first row that is fully NaN.

    Args:
        dataframe (DataFrame): The input pandas DataFrame.

    Returns:
        DataFrame: A new DataFrame with rows removed after and including the first
        fully NaN row. If no such row exists, the original DataFrame is returned.
    """
    # Iterate over the rows and find the first row that is all NaN
    generator = (index for index, row in dataframe.iterrows() if row.isna().all())
    result = next(generator, None)

    if result is not None:
        return dataframe.iloc[: result - 1]  # noqa: Pandas throw bad typing

    return dataframe


def preprocess(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    # File 'input_2.xlsx' refute this method (1/2 full NaN)
    # Drop all next row, when first encounter is all NaN
    dataframe = drop_rows_after_nan(dataframe)

    # Drop column when all values are NaN
    # dataframe = dataframe.dropna(axis=1, how="all")

    return dataframe


def get_helper_output_parser(pydantic_class: Type[BaseModel]):
    """
    Fournit le début du JSON attendus par le parser.

    Args:
        pydantic_class (BaseModel): Modèle Pydantic attendu par le OutputParser

    Notes:
        Cela permet d'éviter les erreurs de génération avec les petits modèles
    """
    first_attr = list(pydantic_class.model_fields)[0]
    return f"""```{{\"{first_attr}\":"""


class Output(BaseModel):
    """
    Output schema that you must respect

    Args:
        reasoning (str): the reasoning behind the choice
        column_ean (str): column name of internal barcode of the product (values of this column is integer)
        column_brand (str): column name of brand name of the product (values of this column is string)
        column_name (str): column name of name of the product (values of this column is string)
        column_quantity (str): column name of quantity of the product (values of this column is integer)
        column_price (str): column name of price of the product (in euros, not in dollars, values of this column is float)
        column_lang (str): column name of country code of product location (values of this column is string)
        column_lt (str): column name of lead time of the product (in days, whole, values of this column is integer)
        column_comment (str): column name of comment of the product (optional) (values of this column is string)
    """

    reasoning: str
    column_ean: Optional[str] = None
    column_brand: Optional[str] = None
    column_name: Optional[str] = None
    column_quantity: Optional[str] = None
    column_price: Optional[str] = None
    column_lang: Optional[str] = None
    column_lt: Optional[str] = None
    column_comment: Optional[str] = None


def find_column(
    dataframe: pandas.DataFrame,
    llm_model: BaseChatModel,
) -> Tuple[Output, int, int]:
    """
    Search which column corresponds to the type of data

    Args:
        dataframe (pandas.DataFrame): the dataframe to process
        llm_model (BaseChatModel): the model to use to find the column

    Returns:
        Tuple[Output, int, int]: the output, the number of input and output token
    """
    query = """\
I'm going to provide you with a Dataframe with columns and examples of data:
- For each attribute of Pydantic class, I'd like you to tell me which column corresponds to this type of data.
- Don't force yourself if you don't know or found the column, just set None in the column name. 
- If column have NaN or empty value, set None in the column name (you can't estimate the type of data).
- DataFrame have to be filled by humans, column name and values can have inconsistency. 
  - For example, 2 columns with 'EAN', 'EAN.1' column name, but 'EAN' defining a Brand and the 'EAN.1' the real EAN, you must give 'EAN' for Brand and 'EAN.1' for EAN.
  - For example 2 columns price but one in $ and the other in €
- Give only the column name not the values."""
    parser = PydanticOutputParser(pydantic_object=Output)
    helper_output_parser = get_helper_output_parser(Output)

    template = """\
Answer the user query
{{format_instructions}}

Here's the Dataframe:
{{dataframe}}

{{query}}

Here is the json corresponding to the pydantic class with the Dataframe you provided
{{helper_output_parser}}"""

    dataframe = dataframe.head(5)
    print(dataframe.to_markdown())

    prompt = PromptTemplate(
        # Todo : Crée un prompt sur LangChain pour générer des tests unitaires.
        template_format="jinja2",
        template=template,
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "helper_output_parser": helper_output_parser,
            "dataframe": dataframe.to_markdown(),
        },
    )

    print(prompt.pretty_repr())

    prompt_and_model = prompt | llm_model
    llm_output = prompt_and_model.invoke({"query": f"{query}"})

    input_tokens = llm_output.response_metadata["token_usage"]["prompt_tokens"]
    output_tokens = llm_output.response_metadata["token_usage"]["completion_tokens"]
    print("Input token:", input_tokens)
    print("Output token:", output_tokens)
    output = parser.parse(llm_output.content)

    return output, input_tokens, output_tokens


def skip_rows(df: pandas.DataFrame, skip_rows: int = 0) -> pandas.DataFrame:
    """
    Replicate 'skip_rows' argument of pandas.read_excel().

    Args:
        df (pd.DataFrame): The input DataFrame.
        skip_rows (int): The number of rows to skip from the top of the DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with rows skipped and column names reset.

    Example:
        data = {'Col1': ['Header1', 'A', 'B', 'C'],
                'Col2': ['Header2', '1', '2', '3'],
                'Col3': ['Header3', 'X', 'Y', 'Z']}
        df = pd.DataFrame(data)
        result_df = skip_rows_and_reset_columns(df, 1)
        print(result_df)
    """

    # Select rows starting from after the skipped rows
    df_skipped = df.iloc[skip_rows:, :].reset_index(drop=True)

    # Reassign columns based on the first row of the new data
    df_skipped.columns = df_skipped.iloc[0]

    # Remove the first row which now serves as the column headers
    df_skipped = df_skipped.iloc[1:, :].reset_index(drop=True)

    return df_skipped


def process_df(
    dataframe: pandas.DataFrame,
    model_id: str = "mistral-large-latest",
    api_key: Optional[str] = None,
) -> Tuple[pandas.DataFrame, int, int]:
    """
    Preprocess Dataframe and find the column a
    Args:
        dataframe (pandas.DataFrame): the dataframe to process
        model_id (str): the model to use to use output parser
        api_key (Optional[str]): the api key to use the model

    Returns:
        Tuple[pandas.DataFrame, int, int]: the dataframe with good format, number of input and output token
    """
    # Load the model
    llm_model = ChatMistralAI(
        model=model_id,
        mistral_api_key=api_key,
        temperature=0.3,
        max_retries=2,
    )

    # Remove blank rows and parasite lines
    skip_row = get_skip_row(dataframe)
    dataframe = skip_rows(dataframe, skip_row)

    # The 'skip_last' variable decreases, so start at the end
    length = len(dataframe.columns)
    dataframe, *_ = strip_unnamed_column(dataframe, increment=1, skip=0)
    dataframe, *_ = strip_unnamed_column(dataframe, increment=-1, skip=length)

    # Remove unnamed columns who are not deleted by strip
    dataframe = split_noname_column(dataframe)
    dataframe = preprocess(dataframe)

    # Get the column name for each data type
    output, input_tokens, output_tokens = find_column(
        dataframe=dataframe,
        llm_model=llm_model,
    )

    # for each out attribute, found the column name, if exist, insert result dataframe
    match_ = {
        "EAN/REF INTERNE": output.column_ean,
        "BRAND": output.column_brand,
        "NOM": output.column_name,
        "QTY": output.column_quantity,
        "PRIX": output.column_price,
        "EXW": output.column_lang,
        "LT": output.column_lt,
        "COMMENT": output.column_comment,
    }

    df_result = pandas.DataFrame()
    for key, value in match_.items():
        conditions = (
            value is not None,
            value in dataframe.columns,
            value != "",
        )

        if all(conditions):
            df_result[key] = dataframe[value]

    return df_result, input_tokens, output_tokens
