import logging
import sys
from colorama import Fore, Style, init
init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        # Use high contrast and bold for WARNING/ERROR/CRITICAL
        if record.levelno >= logging.WARNING:
            prefix = Style.BRIGHT + color
        else:
            prefix = color
        msg = super().format(record)
        return f"{prefix}{msg}{Style.RESET_ALL}"

def get_logger(name: str = "neuro_symbolic_math", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColorFormatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
