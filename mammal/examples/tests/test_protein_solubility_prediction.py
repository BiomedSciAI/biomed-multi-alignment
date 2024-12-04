import socket
from pathlib import Path

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from mammal.examples.protein_solubility.main_infer import protein_solubility_infer
from mammal.main_finetune import main as main_finetune

TEST_CONFIG_DIRPATH = str(Path(__file__).parents[0] / "../protein_solubility")
TEST_CONFIG_FILENAME = "config.yaml"


@pytest.fixture(autouse=True, scope="session")
def _clean_hydra() -> None:
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory):
    if "ccc" not in socket.gethostname():
        pytest.skip("Full tests requires resources")

    model_dir_path = tmp_path_factory.mktemp("test_protein_solubility") / "test"
    return model_dir_path


def test_finetune(model_dir: str):
    print(model_dir)
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        "trainer.max_epochs=2",  # Small number for a faster run.
        "+trainer.limit_train_batches=3",  # Small number for a faster run.
        "+trainer.limit_val_batches=2",  # Small number for a faster run.
        f"model_dir={model_dir}",
        "task.data_module_kwargs.train_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "task.data_module_kwargs.valid_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "root=.",
        "name=sol_test",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    cfg = hydra.utils.instantiate(_cfg)
    main_finetune(cfg)


def test_evaluate(model_dir: str):
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        "trainer.max_epochs=1",  # Small number for a faster run.
        "+trainer.limit_test_batches=10",  # Small number for a faster run.
        f"model_dir={model_dir}",
        "task.data_module_kwargs.train_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "task.data_module_kwargs.valid_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "root=.",
        "name=sol_test",
        "evaluate=True",
        f"model.pretrained_kwargs.pretrained_model_name_or_path={model_dir}/best_epoch.ckpt",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    cfg = hydra.utils.instantiate(_cfg)
    main_finetune(cfg)


def test_infer(model_dir: str):
    protein_solubility_infer(
        finetune_output_dir=model_dir,
        protein_seq="NLMKRCTRGFRKLGKCTTLEEEKCKTLYPRGQCTCSDSKMNTHSCDCKSC",
        device="cpu",
    )
