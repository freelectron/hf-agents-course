import logging
from functools import wraps

import mlflow


# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a package-level logger
logger = logging.getLogger(__name__)
