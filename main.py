# main.py
import logging
from src.utils import setup_logging
from src.advisor import BaccaratLLMAdvisor
from tests.test_scenarios import run_baccarat_tests

if __name__ == "__main__":
    setup_logging()  # Enable logging
    # config_path = "config/llm_advisor_config_qwen.yaml"  # Path to the config file
    config_path = "config/llm_advisor_config_llama_3.yaml"  # Path to the config file
    advisor = BaccaratLLMAdvisor(config_path)  # Initialize advisor with config
    run_baccarat_tests(advisor)  # Run test scenarios