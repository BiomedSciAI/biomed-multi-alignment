import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils import PeftType
from transformers import PreTrainedModel


def get_lora_model(
    model: PreTrainedModel,
    peft_type: str | PeftType | None = None,
    auto_mapping: dict | None = None,
    base_model_name_or_path: str | None = None,
    revision: str | None = None,
    task_type: str | TaskType | None = None,
    inference_mode: bool = False,
    r: int = 8,
    target_modules: list[str] | str | None = None,
    lora_alpha: int = 8,
    lora_dropout: float = 0,
    fan_in_fan_out: bool = False,
    bias: str = "none",
    modules_to_save: list[str] | None = None,
    init_lora_weights: bool = True,
    layers_to_transform: list[int] | None = None,
    layers_pattern: str | None = None,
) -> torch.nn.Module:
    """
    Freeze model params and make lora config. Then convert model to lora.

    Args:
        model (`PreTrainedModel`): The model to convert to Lora.
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    # build lora config
    config = LoraConfig(
        peft_type=peft_type,
        auto_mapping=auto_mapping,
        base_model_name_or_path=base_model_name_or_path,
        revision=revision,
        task_type=task_type,
        inference_mode=inference_mode,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=fan_in_fan_out,
        bias=bias,
        modules_to_save=modules_to_save,
        init_lora_weights=init_lora_weights,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
