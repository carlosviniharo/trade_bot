import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # Customize the output format
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler('app.log', mode='a'),  # Append logs to 'app.log'
        ]
    )