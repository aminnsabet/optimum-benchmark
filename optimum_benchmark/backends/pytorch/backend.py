import gc
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments  # GPTQConfig
from transformers.utils.fx import symbolic_trace

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import TrainerCallback, TrainerState
    from transformers.utils import ModelOutput

from ...profilers.fx_profiler import FXProfilingWrapper
from ..base import Backend
from ..ddp_utils import training_worker
from .config import PyTorchConfig
from .utils import randomize_weights

# bachend logger
LOGGER = getLogger("pytorch")


class PyTorchBackend(Backend[PyTorchConfig]):
    NAME: str = "pytorch"

    def __init__(self, model: str, task: str, device: str, hub_kwargs: Dict[str, Any]):
        super().__init__(model, task, device, hub_kwargs)

        automodel = self.automodel_class.__name__
        LOGGER.info(f"\t+ Infered AutoModel class {automodel} for task {self.task} and model_type {self.model_type}")

    def configure(self, config: PyTorchConfig) -> None:
        super().configure(config)

        # Gradients options
        if self.config.disable_grad:
            LOGGER.info("\t+ Disabling gradients")
            torch.set_grad_enabled(False)

        # Threading options
        if self.config.inter_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch inter_op_num_threads({self.config.inter_op_num_threads}))")
            torch.set_num_threads(self.config.inter_op_num_threads)
        if self.config.intra_op_num_threads is not None:
            LOGGER.info(f"\t+ Setting pytorch intra_op_num_threads({self.config.intra_op_num_threads}))")
            torch.set_num_interop_threads(self.config.intra_op_num_threads)

        # Dtypes options
        self.torch_dtype = getattr(torch, self.config.torch_dtype) if self.config.torch_dtype is not None else None
        self.amp_dtype = getattr(torch, self.config.amp_dtype) if self.config.amp_dtype is not None else None

        # Load model
        if self.config.no_weights:
            self.load_model_from_config()
        else:
            self.load_model_from_pretrained()

        # Eval mode
        if self.config.eval_mode:
            if self.is_diffusion_pipeline():
                LOGGER.info("\t+ Diffusion pipeline are always in eval mode")
            else:
                LOGGER.info("\t+ Turning on model's eval mode")
                self.pretrained_model.eval()

        # BetterTransformer
        if self.config.bettertransformer:
            LOGGER.info("\t+ Using optimum.bettertransformer")
            from optimum.bettertransformer import BetterTransformer

            self.pretrained_model = BetterTransformer.transform(
                self.pretrained_model,
                keep_original_model=False,
            )

        # Compile model
        if self.config.torch_compile:
            if self.is_diffusion_pipeline():
                LOGGER.info("\t+ Using torch.compile on unet forward pass")
                # TODO: should we compile vae and/or clip as well ?
                self.pretrained_model.unet.forward = torch.compile(
                    self.pretrained_model.unet.forward,
                    **self.config.torch_compile_kwargs,
                )
            else:
                LOGGER.info("\t+ Using torch.compile on forward pass")
                self.pretrained_model.forward = torch.compile(
                    self.pretrained_model.forward,
                    **self.config.torch_compile_kwargs,
                )

        if self.config.peft_strategy is not None:
            LOGGER.info("\t+ Applying PEFT")
            from peft import get_peft_model

            from ..peft_utils import get_peft_config_class

            peft_config_class = get_peft_config_class(self.config.peft_strategy)
            peft_config = peft_config_class(**self.config.peft_config)
            self.pretrained_model = get_peft_model(self.pretrained_model, peft_config=peft_config)

    def load_model_from_pretrained(self) -> None:
        if self.config.quantization_strategy == "gptq":
            LOGGER.info("\t+ Processing GPTQ config")
            raise NotImplementedError(
                "Applying GPTQ quantization on pretrained models is not supported yet. "
                "If the model is already quantized, you don't need to specify the quantization strategy."
            )
            # need to process dataset, tokenizer, etc.
            # quantization_config = GPTQConfig(**self.config.quantization_config)
        elif self.config.quantization_strategy == "bnb":
            LOGGER.info("\t+ Processing BnB config")
            quantization_config = BitsAndBytesConfig(**self.config.quantization_config)
        else:
            quantization_config = None

        if self.is_diffusion_pipeline():
            LOGGER.info("\t+ Loading diffusion pipeline")
            self.pretrained_model = self.automodel_class.from_pretrained(
                self.model,
                torch_dtype=self.torch_dtype,
                device_map=self.config.device_map,
                **self.hub_kwargs,
            )
            if self.config.device_map is None:
                LOGGER.info(f"\t+ Moving diffusion pipeline to device: {self.device}")
                # Diffusers does not support loading with torch.device context manager
                self.pretrained_model.to(self.device)
        else:
            if self.config.device_map is not None:
                LOGGER.info(f"\t+ Loading model on visible cuda devices with device_map: {self.config.device_map}")
                self.pretrained_model = self.automodel_class.from_pretrained(
                    self.model,
                    torch_dtype=self.torch_dtype,
                    device_map=self.config.device_map,
                    quantization_config=quantization_config,
                    **self.hub_kwargs,
                )
            else:
                LOGGER.info(f"\t+ Loading model on device: {self.device}")
                with self.device:
                    self.pretrained_model = self.automodel_class.from_pretrained(
                        self.model,
                        torch_dtype=self.torch_dtype,
                        quantization_config=quantization_config,
                        **self.hub_kwargs,
                    )

    def load_model_from_config(self) -> None:
        # TODO: create no_weights tests
        LOGGER.info("\t+ Initializing empty weights model on device: meta")
        from accelerate import init_empty_weights

        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                config=self.pretrained_config,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=self.hub_kwargs.get("trust_remote_code", False),
            )

        if self.config.quantization_strategy is not None:
            LOGGER.info("\t+ Materializing model on cpu for quantization to not OOM")
            self.pretrained_model.to_empty(device="cpu")
            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)
            LOGGER.info("\t+ Processing BnB config")
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

            bnb_quantization_config = BnbQuantizationConfig(
                **self.config.quantization_config,
                torch_dtype=self.config.torch_dtype,
                keep_in_fp32_modules=self.pretrained_model.keep_in_fp32_modules
                if hasattr(self.pretrained_model, "keep_in_fp32_modules")
                else None,
            )
            LOGGER.info("\t+ Quantizing model while on cpu and dispatching to device")
            self.pretrained_model = load_and_quantize_model(
                self.pretrained_model, bnb_quantization_config, device_map=self.config.device_map or self.device
            )
        else:
            LOGGER.info(f"\t+ Materializing model on device: {self.device}")
            self.pretrained_model.to_empty(device=self.device)
            LOGGER.info("\t+ Randomizing model weights")
            randomize_weights(self.pretrained_model)

        LOGGER.info("\t+ Tying weights")
        self.pretrained_model.tie_weights()

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing model for profiling")
        LOGGER.info("\t+ Symbolicly tracing model")
        self.pretrained_model = symbolic_trace(self.pretrained_model, input_names=input_names)
        LOGGER.info("\t+ Wrapping model with FXProfilingWrapper")
        self.pretrained_model = FXProfilingWrapper(self.pretrained_model)

    def forward(self, input: Dict[str, Any], **kwargs) -> "ModelOutput":
        if self.is_diffusion_pipeline():
            return super().forward(input, **kwargs)
        else:
            # TODO: autocast as whole can be managed by one config/kwargs
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
                return super().forward(input, **kwargs)

    def generate(self, input: Dict[str, torch.Tensor], **kwargs) -> "ModelOutput":
        if self.is_diffusion_pipeline():
            return super().generate(input, **kwargs)
        else:
            # TODO: autocast as whole can be managed by one config/kwargs
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.config.amp_autocast):
                return super().generate(input, **kwargs)

    @record
    def train(
        self,
        training_dataset: "Dataset",
        training_arguments: Dict[str, Any],
        training_callbacks: List["TrainerCallback"],
        training_data_collator: Callable,
    ) -> "TrainerState":
        worker_args = (
            "torch",
            LOGGER,
            Trainer,
            TrainingArguments,
            self.config.use_ddp,
            training_dataset,
            training_arguments,
            training_data_collator,
            training_callbacks,
            self.pretrained_model,
        )

        if self.config.use_ddp:
            # For DDP, we log only the state of the first rank as transformers does.
            # since the batch size used in measuring the throughput is the one of world size.
            ddp_config = LaunchConfig(**self.config.ddp_config)
            results = elastic_launch(config=ddp_config, entrypoint=training_worker)(worker_args)[0]
        else:
            # For DP, we can still use training_worker, simply not wrapped by the elastic_launch class.
            results = training_worker(worker_args)

        return results

    def clean(self) -> None:
        super().clean()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()