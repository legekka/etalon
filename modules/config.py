import json
from typing import Optional, Dict, Any


class Config:
    """
    A class to load and manage configuration data from a JSON file or dictionary.

    The Config class provides methods to load configuration parameters from a JSON file or directly from a dictionary,
    and to save the configuration back to a JSON file. It processes and stores various configuration attributes
    needed for setting up and training models.

    Attributes:
        name (str): The name of the configuration or experiment.
        model (str): The model name or path.
        tokenizer (Optional[str]): The tokenizer name or path.
        output_dir (str): The directory to save outputs.
        train_dataset (Union[str, dict]): The training dataset path or configuration.
        eval_dataset_split_size (int): The size of the evaluation dataset split.
        max_seq_length (Optional[int]): The maximum sequence length.
        num_classes (Optional[int]): The number of classes in the dataset.
        batch_size (int): The batch size for training.
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
        num_workers (int): The number of workers for data loading.
        num_epochs (Optional[int]): The number of training epochs.
        max_steps (Optional[int]): The maximum number of training steps.
        learning_rate (float): The initial learning rate.
        d_coef (float): The coefficient for the 'd' parameter.
        eta_min (float): The minimum eta value.
        warmup_steps (int): The number of warmup steps.
        eval_steps (Optional[int]): The number of steps between evaluations.
        save_steps (Optional[int]): The number of steps between saving checkpoints.
        logging_steps (int): The number of steps between logging.
        mlm_probability (float): The masked language modeling probability.
        criterion (str): The loss function to use.
        optimizer (str): The optimizer to use.
        scheduler (str): The scheduler to use.
        wandb (Optional[dict]): The Weights and Biases configuration.

    Methods:
        __init__(config_path=None, json_data=None): Initializes the Config object.
        load(): Loads the configuration parameters from the JSON data.
        save(path): Saves the configuration parameters to a JSON file.
    """

    def __init__(self, config_path: Optional[str] = None, json_data: Optional[Dict[str, Any]] = None):
        """
        Initializes the Config object.

        Args:
            config_path (str, optional): The path to the JSON configuration file.
            json_data (dict, optional): A dictionary containing configuration data.

        Raises:
            ValueError: If neither `config_path` nor `json_data` is provided.
        """
        if config_path is not None:
            with open(config_path, 'r', encoding="utf-8") as f:
                json_data = json.load(f)
            self._json_data = json_data
        elif json_data is not None:
            self._json_data = json_data
        else:
            raise ValueError("Either config_path or json_data must be provided.")

        self.load()

    def load(self):
        """
        Loads the configuration parameters from the JSON data.

        Processes the JSON data and sets the corresponding attributes of the Config object.
        """
        self.name = self._json_data["name"]
        self.model = self._json_data["model"]
        self.tokenizer = self._json_data.get("tokenizer")
        self.output_dir = self._json_data["output_dir"]

        # Process train_dataset
        train_dataset = self._json_data["train_dataset"]
        if isinstance(train_dataset, str):
            self.train_dataset = train_dataset
        elif isinstance(train_dataset, dict):
            self.train_dataset = {
                "path": train_dataset["path"],
                "name": train_dataset["name"],
                "split": train_dataset["split"],
            }
        else:
            raise ValueError("train_dataset must be a string or a dictionary.")

        self.eval_dataset_split_size = self._json_data.get("eval_dataset_split_size", 1000)
        self.max_seq_length = self._json_data.get("max_seq_length")
        self.num_classes = self._json_data.get("num_classes")

        self.batch_size = self._json_data.get("batch_size", 8)
        self.gradient_accumulation_steps = self._json_data.get("gradient_accumulation_steps", 1)
        self.num_workers = self._json_data.get("num_workers", 0)
        self.num_epochs = self._json_data.get("num_epochs")
        self.max_steps = self._json_data.get("max_steps")

        self.learning_rate = self._json_data["learning_rate"]
        self.d_coef = self._json_data.get("d_coef", 1)
        self.eta_min = self._json_data.get("eta_min", 0.0)
        self.warmup_steps = self._json_data.get("warmup_steps", 0)

        self.eval_steps = self._json_data.get("eval_steps")
        self.save_steps = self._json_data.get("save_steps")
        self.logging_steps = self._json_data.get("logging_steps", 5)
        self.mlm_probability = self._json_data.get("mlm_probability", 0.15)

        self.optimizer = self._json_data.get("optimizer", "adamw_torch")
        self.scheduler = self._json_data.get("scheduler", "WarmupThenCosineAnnealingLR")

        # Process wandb configuration
        wandb_config = self._json_data.get("wandb")
        if wandb_config:
            self.wandb = {
                "project": wandb_config["project"],
                "name": wandb_config.get("name"),
                "tags": wandb_config.get("tags"),
            }
        else:
            self.wandb = None

    def save(self, path: str):
        """
        Saves the configuration parameters to a JSON file.

        Args:
            path (str): The file path where the configuration should be saved.
        """
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self._json_data, f, ensure_ascii=False, indent=4)
