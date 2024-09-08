import functools
import logging
from typing import Tuple, Optional

import gradio as gr
import pandas
import pandas as pd
import typer

from fbc_process.process import process_df

PRICE_1M_INPUT_TOKENS = 2.7
PRICE_1M_OUTPUT_TOKENS = 8.2


def estimate_price(input_token: int, output_token: int) -> float:
    """
    Estimate the price for the process.

    Args:
        input_token (int): The number of input tokens.
        output_token (int): The number of output tokens.

    Returns:
        float: The estimated price.
    """
    calcul_input = (input_token / 1_000_000) * PRICE_1M_INPUT_TOKENS
    calcul_output = (output_token / 1_000_000) * PRICE_1M_OUTPUT_TOKENS
    calcul = calcul_input + calcul_output

    return calcul


def _gradio_process(
        list_path: list,
        model_id: str = "mistral-large-latest",
        api_key: Optional[str] = None,
) -> Tuple[pandas.DataFrame, int, int]:
    """
    Pipeline Gradio for process Data and return metadata

    Args:
        list_path (list): List of paths to the Excel files to process.

    Returns:
        pandas.DataFrame: The resulting DataFrame.
    """
    total_input_token = 0
    total_output_token = 0
    length = len(list_path)
    df_final = pd.DataFrame()

    for path in list_path:
        df = pandas.read_excel(path)
        output = process_df(df, model_id=model_id, api_key=api_key)
        df_result, input_token, output_token = output

        df_final = pd.concat([df_final, df_result], ignore_index=True)
        total_input_token += input_token
        total_output_token += output_token

    calcul = estimate_price(total_input_token, total_output_token)
    logging.info(f"Estimated price: {calcul:.8f} €")

    calcul_nbr_request = (1 / calcul) / length
    logging.info(f"Number request for 1€ (by file): {calcul_nbr_request:.2f}")

    return df_final, total_input_token, total_output_token


def app(
        debug: bool = typer.Option(False, envvar="DEBUG", help="Enable the debug mode."),
        host: str = typer.Option("127.0.0.1", envvar="HOST", help="Adress to listen on."),
        port: int = typer.Option(7860, envvar="PORT", help="Port to listen on."),
        ssl_keyfile: str = typer.Option(None, envvar="SSL_KEYFILE", help="File containing the SSL key."),
        ssl_certfile: str = typer.Option(None, envvar="SSL_CERTFILE", help="File containing the SSL certificate."),
        model_id: str = typer.Option("mistral-large-latest", help="Model Mistral to use."),
        mistral_token: str = typer.Option(None, envvar="MISTRAL_API_KEY", help="Token to access the Mistral API."),
        max_file_size: str = typer.Option("10mb", envvar="MAX_FILE_SIZE", help="Size of the maximum file to download."),
        enable_monitoring: bool = typer.Option(True, envvar="ENABLE_MONITORING",
                                               help="Activate the monitoring of the application."),
):
    """
    Start the Gradio server to serve the model.

    Args:
        debug (bool): Enable the debug mode.
        host (str): Adress to listen on.
        port (int): Port to listen on.
        ssl_keyfile (str): File containing the SSL key.
        ssl_certfile (str): File containing the SSL certificate.
        mistral_token (str): Token to access the Mistral API.
        model_id (str): Model Mistral to use.
        max_file_size (str): Size of the maximum file to download.
        enable_monitoring (bool): Activate the monitoring of the application.
    """
    logging.debug("Starting the Gradio application")
    gradio_process = functools.partial(
        _gradio_process,
        model_id=model_id,
        api_key=mistral_token,
    )

    with gr.Blocks() as application:
        gr.Markdown("## FBC Process")
        gr.Markdown(
            "This application detects the format of Excel files, reformats them and renames columns according to their data type.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    file_count="multiple",
                    label="Upload your Excel file",
                    file_types=[
                        ".xlsx",
                    ],
                )
                button = gr.Button()
                info_input_token = gr.Textbox("0", label="Input token")
                info_output_token = gr.Textbox("0", label="Output token")

            with gr.Column(scale=4):
                dataframe = gr.DataFrame()

        button.click(
            fn=gradio_process,
            inputs=file_input,
            outputs=[
                dataframe,
                info_input_token,
                info_output_token
            ]
        )

    application.launch(
        ssl_verify=False,
        debug=debug,
        server_name=host,
        server_port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        max_file_size=max_file_size,
        enable_monitoring=enable_monitoring,
    )


if __name__ == "__main__":
    app()
