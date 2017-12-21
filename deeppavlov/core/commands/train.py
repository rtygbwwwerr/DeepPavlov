from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.commands.infer import build_agent_from_config
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common import paths


# TODO pass paths to local model configs to agent config.


def train_agent_models(config_path: str):
    usr_dir = paths.USR_PATH
    a = build_agent_from_config(config_path)

    for skill_config in a.skill_configs:
        model_config = skill_config['model']
        model_name = model_config['name']

        if issubclass(_REGISTRY[model_name], Trainable):
            reader_config = skill_config['dataset_reader']
            reader = from_params(_REGISTRY[reader_config['name']], {})
            data = reader.read(reader_config.get('data_path', usr_dir))

            dataset_config = skill_config['dataset']
            dataset_name = dataset_config['name']
            dataset = from_params(_REGISTRY[dataset_name], dataset_config, data=data)

            model = from_params(_REGISTRY[model_name], model_config)
            model.train(dataset)
        else:
            print('Model {} is not an instance of Trainable, skip training.'.format(model_name))


def train_model_from_config(config_path: str):
    usr_dir = paths.USR_PATH
    config = read_json(config_path)

    reader_config = config['dataset_reader']
    reader = from_params(_REGISTRY[reader_config['name']], {})
    data = reader.read(reader_config.get('data_path', usr_dir))

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset = from_params(_REGISTRY[dataset_name], dataset_config, data=data)

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(_REGISTRY[model_name], model_config)

    model.train(dataset)

    # The result is a saved to user_dir trained model.
