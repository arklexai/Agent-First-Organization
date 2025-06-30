"""Model utilities for the TAU benchmark.

This module provides utilities for working with language models in the TAU benchmark,
including API interfaces, model implementations, and data handling. It supports
various model types (OpenAI, etc.) and provides functionality for classification,
generation, parsing, and scoring tasks.

The module includes:
- API interfaces for model interaction
- Model implementations for different platforms
- Data point handling and evaluation
- Sampling strategies for model outputs
- Error handling and exceptions
"""

from benchmark.tau_bench.model_utils.api.api import (
    BinaryClassifyDatapoint as BinaryClassifyDatapoint,
)
from benchmark.tau_bench.model_utils.api.api import (
    ClassifyDatapoint as ClassifyDatapoint,
)
from benchmark.tau_bench.model_utils.api.api import (
    GenerateDatapoint as GenerateDatapoint,
)
from benchmark.tau_bench.model_utils.api.api import (
    ParseForceDatapoint as ParseForceDatapoint,
)
from benchmark.tau_bench.model_utils.api.api import (
    default_api_from_args as default_api_from_args,
)
from benchmark.tau_bench.model_utils.api.api import (
    default_quick_api as default_quick_api,
)
from benchmark.tau_bench.model_utils.api.datapoint import (
    EvaluationResult as EvaluationResult,
)
from benchmark.tau_bench.model_utils.api.datapoint import (
    datapoint_factory as datapoint_factory,
)
from benchmark.tau_bench.model_utils.api.datapoint import (
    load_from_disk as load_from_disk,
)
from benchmark.tau_bench.model_utils.api.sample import (
    EnsembleSamplingStrategy as EnsembleSamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    MajoritySamplingStrategy as MajoritySamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    RedundantSamplingStrategy as RedundantSamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    RetrySamplingStrategy as RetrySamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    SamplingStrategy as SamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    SingleSamplingStrategy as SingleSamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    UnanimousSamplingStrategy as UnanimousSamplingStrategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    get_default_sampling_strategy as get_default_sampling_strategy,
)
from benchmark.tau_bench.model_utils.api.sample import (
    set_default_sampling_strategy as set_default_sampling_strategy,
)
from benchmark.tau_bench.model_utils.model.chat import (
    PromptSuffixStrategy as PromptSuffixStrategy,
)
from benchmark.tau_bench.model_utils.model.general_model import (
    GeneralModel as GeneralModel,
)
from benchmark.tau_bench.model_utils.model.general_model import (
    default_model as default_model,
)
from benchmark.tau_bench.model_utils.model.general_model import (
    model_factory as model_factory,
)
from benchmark.tau_bench.model_utils.model.model import (
    BinaryClassifyModel as BinaryClassifyModel,
)
from benchmark.tau_bench.model_utils.model.model import (
    ParseForceModel as ParseForceModel,
)
