import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config_benchmark.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.utils import utils
    from src.benchmark import benchmark
    
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    
    return benchmark(config) 


if __name__ == "__main__":
    main()
