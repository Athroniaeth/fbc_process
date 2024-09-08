# Utiliser une image Python avec Poetry
FROM python:3.12-slim

# Installer Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy dependencies files
COPY pyproject.toml poetry.lock ./

# Install the dependencies
RUN poetry install --no-dev

# Copy the .env file
COPY .env .

# Copy the source files
COPY src/ src/

# Expose the port 7860
EXPOSE 7862

# Set the command to run the application
CMD ["poetry", "run", "python", "src/fbc_process", "--host", "0.0.0.0", "--port", "7862", "--ssl-keyfile", "/etc/letsencrypt/live/pierrechaumont.fr/privkey.pem", "--ssl-certfile", "/etc/letsencrypt/live/pierrechaumont.fr/fullchain.pem"]
