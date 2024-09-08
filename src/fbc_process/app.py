import logging
from typing import Tuple

import gradio as gr
import pandas
import pandas as pd
import typer


def process_df(path: str) -> pandas.DataFrame:
    return pandas.DataFrame()


def gradio_process(list_path: list) -> Tuple[pandas.DataFrame, int, int]:
    """
    Pipeline Gradio for process Data and return metadata

    Args:
        list_path (list): List of paths to the Excel files to process.

    Returns:
        pandas.DataFrame: The resulting DataFrame.
    """
    df_final = pd.DataFrame()

    for path in list_path:
        df = process_df(path)
        df_final = pd.concat([df_final, df], ignore_index=True)

    return df_final


def app(
        debug: bool = typer.Option(False, envvar="DEBUG", help="Enable the debug mode."),
        host: str = typer.Option("127.0.0.1", envvar="HOST", help="Adress to listen on."),
        port: int = typer.Option(7860, envvar="PORT", help="Port to listen on."),
        ssl_keyfile: str = typer.Option(None, envvar="SSL_KEYFILE", help="File containing the SSL key."),
        ssl_certfile: str = typer.Option(None, envvar="SSL_CERTFILE", help="File containing the SSL certificate."),
        mistral_token: str = typer.Option(None, envvar="MISTRAL_API_KEY", help="Token to access the Mistral API."),
        model_id: str = typer.Option("mistral-large-latest", help="Model Mistral to use."),
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

    with gr.Blocks() as application:
        file_input = gr.File(
            file_count="multiple",
            label="Upload your Excel file",
            file_types=[
                ".xlsx",
            ],
        )
        button = gr.Button()
        dataframe = gr.DataFrame()

        button.click(fn=gradio_process, inputs=file_input, outputs=dataframe)

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
