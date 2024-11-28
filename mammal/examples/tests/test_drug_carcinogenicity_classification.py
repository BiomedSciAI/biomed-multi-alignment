from pathlib import Path

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from mammal.examples.carcinogenicity.main_infer import infer
from mammal.main_finetune import main as main_finetune
from mammal.model import Mammal

TEST_CONFIG_DIRPATH = str(Path(__file__).parents[0] / "../carcinogenicity")
TEST_CONFIG_FILENAME = "config.yaml"


@pytest.fixture(autouse=True, scope="session")
def _clean_hydra() -> None:
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session")
def tmp_model_dir(tmp_path_factory):
    model_dir_path = tmp_path_factory.mktemp("test_carcinogenicity") / "test"
    return model_dir_path


@pytest.fixture(scope="session")
def finetuned_model_dir(tmp_model_dir: str):
    model_dir = tmp_model_dir
    print(f"\n{model_dir=}")
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        "trainer.max_epochs=2",  # Small number for a faster run.
        "+trainer.limit_train_batches=3",  # Small number for a faster run.
        "+trainer.limit_val_batches=2",  # Small number for a faster run.
        f"model_dir={model_dir}",
        "root=.",
        "name=carcinogenicity_test",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    main_finetune(_cfg)
    return model_dir


def test_finetune(finetuned_model_dir: Path):
    # the actual work is done in the fixture, here we just read the finetuned model.
    model = Mammal.from_pretrained(
        pretrained_model_name_or_path=str(finetuned_model_dir / "best_epoch.ckpt")
    )
    assert model is not None


def test_evaluate(finetuned_model_dir: str):
    OVERRIDES = [
        "track_clearml=null",  # Travis cannot connect to ClearML at the moment. We might be able to fix it with a dedicated user + config credentials.
        "trainer.max_epochs=1",  # Small number for a faster run.
        "+trainer.limit_test_batches=10",  # Small number for a faster run.
        f"model_dir={finetuned_model_dir}",
        "root=.",
        "name=carcinogenicity_test",
        "evaluate=True",
        f"model.pretrained_kwargs.pretrained_model_name_or_path={finetuned_model_dir}/best_epoch.ckpt",
    ]
    with hydra.initialize_config_dir(TEST_CONFIG_DIRPATH, version_base="1.1"):
        _cfg = hydra.compose(TEST_CONFIG_FILENAME, overrides=OVERRIDES)
    # main_finetune does "evaluate" if evalute=True, as is set above.
    main_finetune(_cfg)


def test_infer(finetuned_model_dir: str):
    infer(
        finetune_output_dir=finetuned_model_dir,
        drug_seq="CC(CCl)OC(C)CCl",
        device="cpu",
    )
