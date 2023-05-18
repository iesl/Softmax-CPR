# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, has_length, denumpify_detensorize, speed_metrics
from transformers.utils import is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size, nested_concat, nested_detach, nested_numpify, nested_truncate
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
#     Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

class Seq2SeqTrainer_updated(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

#         # XXX: adapt synced_gpus for fairscale as well
#         gen_kwargs = {
#             "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
#             "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
#             #"synced_gpus": True if is_deepspeed_zero3_enabled() else False, #delete Haw-Shiuan
#         }
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

#         generated_tokens = self.model.generate(
#             generation_inputs,
#             do_sample=True, #ADD Haw-Shiuan
#             **gen_kwargs,
#         )
        
        generated_tokens = self.model.generate(
            generation_inputs,
            do_sample=True, 
            max_length=128, 
            num_beams=1, 
            top_k=10
        )
        
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)