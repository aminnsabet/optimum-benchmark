import os
from itertools import product
from logging import getLogger

from llm_perf.constants import CANONICAL_MODELS_LIST, GENERATE_KWARGS, INPUT_SHAPES, PRETRAINED_MODELS_LIST
from llm_perf.utils import common_errors_reporter, is_experiment_conducted, is_experiment_not_supported
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.logging_utils import setup_logging

CWD = os.getcwd()
MACHINE = os.getenv("MACHINE", "1xA100")
SUBSET = os.getenv("SUBSET", "unquantized")
CANONICAL_MODELS_ONLY = os.getenv("CANONICAL_MODELS_ONLY", "1") == "1"
PUSH_REPO_ID = f"optimum-benchmark/llm-perf-pytorch-cuda-{SUBSET}-{MACHINE}"


ATTENTION_COFIGS = ["eager", "sdpa", "flash_attention_2"]
if SUBSET == "unquantized":
    WEIGHTS_CONFIGS = {
        # unquantized
        "float32": {"torch_dtype": "float32", "quant_scheme": None, "quant_config": {}},
        "float16": {"torch_dtype": "float16", "quant_scheme": None, "quant_config": {}},
        "bfloat16": {"torch_dtype": "bfloat16", "quant_scheme": None, "quant_config": {}},
    }
elif SUBSET == "bnb":
    WEIGHTS_CONFIGS = {
        # bnb
        "4bit-bnb": {"torch_dtype": "float16", "quant_scheme": "bnb", "quant_config": {"load_in_4bit": True}},
        "8bit-bnb": {"torch_dtype": "float16", "quant_scheme": "bnb", "quant_config": {"load_in_8bit": True}},
    }
elif SUBSET == "gptq":
    WEIGHTS_CONFIGS = {
        # gptq
        "4bit-gptq-exllama-v1": {
            "quant_scheme": "gptq",
            "torch_dtype": "float16",
            "quant_config": {"bits": 4, "use_exllama ": True, "version": 1, "model_seqlen": 256},
        },
        "4bit-gptq-exllama-v2": {
            "torch_dtype": "float16",
            "quant_scheme": "gptq",
            "quant_config": {"bits": 4, "use_exllama ": True, "version": 2, "model_seqlen": 256},
        },
    }
elif SUBSET == "awq":
    WEIGHTS_CONFIGS = {
        # awq
        "4bit-awq-gemm": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {"bits": 4, "version": "gemm"},
        },
        "4bit-awq-gemv": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {"bits": 4, "version": "gemv"},
        },
        "4bit-awq-exllama-v1": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {
                "bits": 4,
                "version": "exllama",
                "exllama_config": {"version": 1, "max_input_len": 64, "max_batch_size": 1},
            },
        },
        "4bit-awq-exllama-v2": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {
                "bits": 4,
                "version": "exllama",
                "exllama_config": {"version": 2, "max_input_len": 64, "max_batch_size": 1},
            },
        },
    }


setup_logging()
LOGGER = getLogger("llm-perf-backend")


def benchmark_cuda_pytorch(model, attn_implementation, weights_config):
    torch_dtype = WEIGHTS_CONFIGS[weights_config]["torch_dtype"]
    quant_scheme = WEIGHTS_CONFIGS[weights_config]["quant_scheme"]
    quant_config = WEIGHTS_CONFIGS[weights_config]["quant_config"]

    if is_experiment_not_supported(torch_dtype, attn_implementation):
        LOGGER.info(f"Skipping experiment with model {model} since it is not supported")
        return

    launcher_config = ProcessConfig(
        start_method="spawn",
        device_isolation=True,
        device_isolation_action="error",
    )
    benchmark_config = InferenceConfig(
        memory=True,
        energy=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes=INPUT_SHAPES,
        generate_kwargs=GENERATE_KWARGS,
    )
    backend_config = PyTorchConfig(
        model=model,
        device="cuda",
        device_ids="0",
        no_weights=True,
        library="transformers",
        task="text-generation",
        torch_dtype=torch_dtype,
        quantization_scheme=quant_scheme,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
    )

    experiment_name = f"{weights_config}-{attn_implementation}"
    subfolder = f"{experiment_name}/{model.replace('/', '--')}"

    experiment_config = ExperimentConfig(
        experiment_name=experiment_name,
        benchmark=benchmark_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    if is_experiment_conducted(experiment_config, PUSH_REPO_ID, subfolder):
        LOGGER.info(f"Skipping experiment {experiment_name} with model {model} since it was already conducted")
        return

    experiment_config.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)

    try:
        benchmark_report = launch(experiment_config)
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)
    except Exception as error:
        os.chdir(CWD)  # TODO: figure our why this is happening
        LOGGER.error(f"Experiment {experiment_name} failed with model {model}")
        common_errors_reporter(error, LOGGER, subfolder, PUSH_REPO_ID)


if __name__ == "__main__":
    if CANONICAL_MODELS_ONLY:
        models_attentions_weights = list(product(CANONICAL_MODELS_LIST, ATTENTION_COFIGS, WEIGHTS_CONFIGS.keys()))
        print(f"Total number of canonical models experiments: {len(models_attentions_weights)}")
    else:
        models_attentions_weights = list(product(PRETRAINED_MODELS_LIST, ATTENTION_COFIGS, WEIGHTS_CONFIGS.keys()))
        print(f"Total number of pretrained models experiments: {len(models_attentions_weights)}")

    for model, attn_implementation, weights_config in models_attentions_weights:
        benchmark_cuda_pytorch(model, attn_implementation, weights_config)
