import os
from collections.abc import Callable
from functools import partial

import hydra
import pytorch_lightning as pl
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.utils import NDict
from omegaconf import DictConfig, OmegaConf

from mammal.model import Mammal
from mammal.task import MammalTask


def save_in_model_dir(
    model_dir: str, model: Mammal, tokenizer_op: ModularTokenizerOp
) -> None:
    """
    Save model configuration and tokenizer in model_dir before starting the finetunning session
    :param model_dir: location to store the files
    :param model: the model to save the configuration for
    :param tokenizer_op: the tokenizer to save
    """

    if model_dir is None:
        return

    os.makedirs(model_dir, exist_ok=True)

    # save model config
    model._save_pretrained(model_dir, save_config_only=True)

    # tokenizer
    tokenizer_op.save_pretrained(os.path.join(model_dir, "tokenizer"))


def configure_optimizers(
    module: LightningModuleDefault, opt_callable: Callable, lr_sch_callable: Callable
) -> dict:
    """
    A callback use by lightning module to set the learning rate scheduler and the optimizer
    :param module: the lightning module
    :param opt_callable: a callable that creates an optimizer given the model parameters
    :param lr_sch_callable: a callable that creates a learning rate scheduler given the optimizer

    """
    opt = opt_callable(module.trainer.model.parameters())
    lr_sch = lr_sch_callable(opt)
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": lr_sch, "interval": "step"},
    }


def module(
    model: Mammal,
    task: MammalTask,
    opt_callable: Callable,
    lr_sch_callable: Callable,
    **kwargs,
) -> pl.LightningModule:
    """
    Create lightning module
    :param task: the task to finetune for
    :param opt_callable: a callable that creates an optimizer given the model parameters
    :param lr_sch_callable: a callable that creates a learning rate scheduler given the optimizer
    :param kwargs: additional LightningModuleDefault arguments
    """
    optimizers_and_lr_schs_callable = partial(
        configure_optimizers, opt_callable=opt_callable, lr_sch_callable=lr_sch_callable
    )
    return LightningModuleDefault(
        model=model,
        losses=task.losses(),
        validation_metrics=task.validation_metrics(),
        train_metrics=task.train_metrics(),
        optimizers_and_lr_schs=optimizers_and_lr_schs_callable,
        **kwargs,
    )


@hydra.main(version_base="1.2", config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)

    # print configuration
    NDict(OmegaConf.to_container(cfg, resolve=True)).print_tree(True)

    # connect to clearml - if configured
    if "track_clearml" in cfg and cfg["track_clearml"] is not None:
        try:
            from fuse.dl.lightning.pl_funcs import start_clearml_logger

            clearml_task = start_clearml_logger(**cfg.track_clearml)
        except Exception as e:
            print("Tracking using clearml failed: continue without tracking")
            print(e)
            clearml_task = None

        if clearml_task is not None:
            clearml_logger = clearml_task.get_logger()
        else:
            clearml_logger = None  # will be None in dist training and rank != 0
    else:
        clearml_logger = None

    # seed
    pl.seed_everything(seed=cfg.seed, workers=True)

    # tokenizer
    tokenizer_op = load_and_update_tokenizer_op(cfg)

    # model
    model = Mammal.from_pretrained(**cfg.model.pretrained_kwargs)
    print(model)

    # initialize task
    task: MammalTask = cfg.task(tokenizer_op=tokenizer_op, logger=clearml_logger)

    # lightning data module
    pl_data_module = task.data_module()

    # lightning module
    pl_module = module(
        task=task,
        model=model,
        **OmegaConf.to_container(cfg.module, resolve=True),
    )

    # create lightning trainer.
    pl_trainer = pl.Trainer(**cfg.trainer)

    if cfg.evaluate:
        pl_data_module.setup("test")
        out = pl_trainer.test(
            model=pl_module,
            dataloaders=pl_data_module.test_dataloader(),
        )
        print(out)
    else:
        # save model_config and tokenizer in output_dir
        save_in_model_dir(cfg.module.model_dir, model, tokenizer_op)
        pl_trainer.fit(model=pl_module, datamodule=pl_data_module)


def load_and_update_tokenizer_op(cfg):
    tokenizer_op = ModularTokenizerOp.from_pretrained(cfg.tokenizer.tokenizer_path)

    if "new_special_tokens" in cfg.tokenizer and len(cfg.tokenizer.new_special_tokens):
        num_new_tokens_added = tokenizer_op.add_new_special_tokens(
            cfg.tokenizer.new_special_tokens
        )
        if num_new_tokens_added:
            # TODO: write better message
            print(
                10 * "****",
                f" Added { num_new_tokens_added} special tokens to the tokenizer ",
                10 * "****",
            )

    return tokenizer_op


if __name__ == "__main__":
    main()
