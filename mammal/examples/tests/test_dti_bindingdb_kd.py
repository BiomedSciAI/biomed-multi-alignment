import socket
from pathlib import Path

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from mammal.examples.dti_bindingdb_kd.main_infer import dti_bindingdb_kd_infer
from mammal.main_finetune import main as main_finetune

TEST_CONFIG_DIRPATH = str(Path(__file__).parents[0] / "../dti_bindingdb_kd")
TEST_CONFIG_FILENAME = "config.yaml"


@pytest.fixture(autouse=True, scope="session")
def _clean_hydra() -> None:
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    if "ccc" not in socket.gethostname():
        pytest.skip("Full tests requires resources")

    model_dir_path = tmp_path_factory.mktemp("test_dti_bindingdb_kd") / "test"
    return model_dir_path


@pytest.mark.xfail(reason="tokenizer only available on the CCC for now")
def test_finetune(model_dir: str) -> None:
    print(model_dir)
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        "trainer.accelerator=auto",
        "trainer.max_epochs=2",  # Small number for a faster run.
        "+trainer.limit_train_batches=3",  # Small number for a faster run. This is not applicable for validation mb per epoch! (WHY?)
        "+trainer.limit_val_batches=2",  # Small number for a faster run. This is not applicable for validation mb per epoch! (WHY?)
        "task.data_module_kwargs.batch_size=1",
        f"model_dir={model_dir}",
        "task.data_module_kwargs.train_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "task.data_module_kwargs.valid_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "root=.",
        "name=dti_bindingdb_kd_test",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    cfg = hydra.utils.instantiate(_cfg)
    main_finetune(cfg)


@pytest.mark.xfail(reason="tokenizer only available on the CCC for now")
def test_evaluate(model_dir: str):
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        # "train.trainer.accelerator=cpu",  # Travis doesn't have GPU.
        "trainer.accelerator=auto",
        "trainer.max_epochs=1",
        "task.data_module_kwargs.batch_size=1",
        "+trainer.limit_test_batches=10",  # Small number for a faster run. This is not applicable for validation mb per epoch! (WHY?)
        f"model_dir={model_dir}",
        "task.data_module_kwargs.train_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "task.data_module_kwargs.valid_dl_kwargs.num_workers=0",  # Using parallelization cause co
        "root=.",
        "name=dti_bindingdb_kd_test",
        "evaluate=True",
        f"model.pretrained_kwargs.pretrained_model_name_or_path={model_dir}/best_epoch.ckpt",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    cfg = hydra.utils.instantiate(_cfg)
    main_finetune(cfg)


@pytest.mark.xfail(reason="tokenizer only available on the CCC for now")
def test_infer(model_dir: str):
    dti_bindingdb_kd_infer(
        finetune_output_dir=model_dir,
        target_seq="NLMKRCTRGFRKLGKCTTLEEEKCKTLYPRGQCTCSDSKMNTHSCDCKSC",
        drug_seq="CC(=O)NCCC1=CNc2c1cc(OC)cc2",
        norm_y_mean=0.0,
        norm_y_std=1.0,
        device="cpu",
    )
