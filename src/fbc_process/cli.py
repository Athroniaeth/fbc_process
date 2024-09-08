import logging
import os

import typer

from fbc_process.app import app

cli = typer.Typer()


@cli.command()
def run(
    host: str = typer.Option("127.0.0.1", envvar="HOST", help="Adress to listen on."),
    port: int = typer.Option(7860, envvar="PORT", help="Port to listen on."),
    ssl_keyfile: str = typer.Option(None, envvar="SSL_KEYFILE", help="File containing the SSL key."),
    ssl_certfile: str = typer.Option(None, envvar="SSL_CERTFILE", help="File containing the SSL certificate."),
    mistral_token: str = typer.Option(None, envvar="MISTRAL_API_KEY", help="Token to access the Mistral API."),
    model_id: str = typer.Option("mistral-large-latest", help="Model Mistral to use."),
    max_file_size: str = typer.Option("10mb", envvar="MAX_FILE_SIZE", help="Size of the maximum file to download."),
    enable_monitoring: bool = typer.Option(True, envvar="ENABLE_MONITORING", help="Activate the monitoring of the application."),
):
    """
    Start the Gradio server to serve the model.

    Args:
        host (str): The address on which the server should listen.
        port (int): The port on which the server should listen.
        ssl_keyfile (str): The SSL key file.
        ssl_certfile (str): The SSL certificate file.
        model_id (str): The HuggingFace model LLM identifier.
        max_file_size (str): The maximum file size to download.
        enable_monitoring (bool): Enable the monitoring of the application.
    """
    # Check if SSL key and certificate files exist
    if ssl_keyfile and not os.path.exists(ssl_keyfile):
        raise FileNotFoundError(f"SSL Key file '{ssl_keyfile}' not found")

    if ssl_certfile and not os.path.exists(ssl_certfile):
        raise FileNotFoundError(f"SSL Certificate file '{ssl_certfile}' not found")

    if (not ssl_keyfile and ssl_certfile) or (ssl_keyfile and not ssl_certfile):
        raise ValueError(f"Both SSL Key and Certificate files must be provided ({ssl_keyfile=}, {ssl_certfile=})")

    # Log the environment information
    ssl = bool(ssl_keyfile and ssl_certfile)
    logging.info(f"{host=}, {port=}, {ssl=}")

    app(
        model_id=model_id,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        mistral_token=mistral_token,
        max_file_size=max_file_size,
        enable_monitoring=enable_monitoring,
    )


if __name__ == "__main__":
    cli()
