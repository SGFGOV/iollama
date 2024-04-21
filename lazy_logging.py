import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def lazy_log(level, message, *args, **kwargs):
    """
    A lazy logging function that only formats the message if the logging level is appropriate.
    
    :param level: The logging level (e.g., logging.INFO, logging.DEBUG).
    :param message: The log message format string.
    :param args: Positional arguments to be formatted into the message string.
    :param kwargs: Keyword arguments to be formatted into the message string.
    """
    if logging.getLogger().isEnabledFor(level):
        # If the logging level is appropriate, log the message
        logging.log(level, message, *args, **kwargs)

# Example usage
lazy_log(logging.DEBUG, "This is a debug message with %s and %d", "string", 123)
lazy_log(logging.INFO, "This is an info message with %s and %d", "string", 123)
