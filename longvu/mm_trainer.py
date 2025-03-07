# pyre-strict
import os
import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import torch
import torch.nn as nn

#### DECODE TRANSFORMERS ####
from transformers.utils import logging
# Create a logger consistent with Hugging Face
logger = logging.get_logger(__name__)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
#### END DECODE TRANSFORMERS ####

from longvu.mm_datautils import get_mm_adapter_state_maybe_zero_3
from torch.utils.data import DataLoader, Sampler
from constants import IMAGE_TOKEN_INDEX
import copy

from transformers import Trainer

from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, has_length, _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging
from resource_logging import *
tcm_logger = logging.getLogger("tcm_logger")

class EvalPredictionWithMask(EvalPrediction):
    
    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        masks: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        super().__init__(predictions, label_ids, inputs)
        self.masks = masks

    def __iter__(self):
        if self.inputs is not None and self.masks is not None:
            return iter((self.predictions, self.label_ids, self.inputs, self.masks))
        elif self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        elif self.masks is not None:
            return iter((self.predictions, self.label_ids, self.masks))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 3:
            raise IndexError("tuple index out of range")
        if idx == 3 and self.masks is None:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None and self.masks is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            if self.inputs is None:
                return self.masks
            else:
                return self.inputs
        elif idx == 3:
            return self.masks


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


def get_mm_adapter_state_maybe_zero_3(
    named_params: Dict[str, torch.Tensor], keys_to_match: List[str]
) -> Dict[str, torch.Tensor]:
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()  # pyre-ignore
        for k, v in to_return.items()
    }
    return to_return


def maybe_zero_3(
    param: torch.Tensor, ignore_status: bool = False, name: Optional[str] = None
) -> torch.Tensor:
    return param.detach().cpu().clone()


class LLaVATrainer(Trainer):
    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        deepspeed: Optional[str] = None,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.snapshot_counter = 0
        self.deepspeed = deepspeed,
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataloader is not None:
            print("Using sonic dataloader")
            # pyre-fixme[16]: `LLaVATrainer` has no attribute `accelerator`.
            return self.accelerator.prepare(self.train_dataloader)
        # pyre-fixme[16]: `Trainer` has no attribute `get_train_dataloader`.
        return super().get_train_dataloader()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # pyre-fixme[16]: `LLaVATrainer` has no attribute `train_dataset`.
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # pyre-fixme[16]: `LLaVATrainer` has no attribute `args`.
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            # pyre-fixme[16]: `Trainer` has no attribute `_get_train_sampler`.
            return super()._get_train_sampler()

    # pyre-fixme[3]: Return type must be annotated.
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # pyre-fixme[16]: `Trainer` has no attribute `model`.
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        # pyre-fixme[16]: `Trainer` has no attribute `optimizer`.
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # pyre-fixme[16]: `Trainer` has no attribute `mm_projector_lr`.
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.mm_vision_sampler_lr is not None:
                vision_sampler_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if ("vision_sampler" in name) or ("vision_query" in name)
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in vision_sampler_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif (
                self.args.unfreeze_mm_vision_tower
                and self.args.mm_vision_tower_lr is not None
            ):
                vision_tower_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "vision_tower" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
        return self.optimizer

    # pyre-fixme[2]: Parameter must be annotated.
    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        # pyre-fixme[16]: `LLaVATrainer` has no attribute `args`.
        if getattr(self.args, "tune_mm_mlp_adapter", False):

            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            # pyre-fixme[16]: `LLaVATrainer` has no attribute `state`.
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            # pyre-fixme[16]: `LLaVATrainer` has no attribute `_get_output_dir`.
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                # pyre-fixme[16]: `LLaVATrainer` has no attribute `model`.
                self.model.named_parameters(),
                keys_to_match,
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            # pyre-fixme[16]: `Trainer` has no attribute `_save_checkpoint`.
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    # pyre-fixme[2]: Parameter must be annotated.
    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        # pyre-fixme[16]: `LLaVATrainer` has no attribute `args`.
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            # pyre-fixme[16]: `Trainer` has no attribute `_save`.
            super(LLaVATrainer, self)._save(output_dir, state_dict)

#    def compute_loss(self, model, inputs, return_outputs=False):
#        """
#        How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#        Subclass and override for custom behavior.
#        """
#        if "labels" in inputs and isinstance(inputs["labels"], torch.Tensor):
#            tmp_labels = copy.deepcopy(inputs["labels"])
#            debug_tensor("In compute_loss(): inputs['labels']", tmp_labels)
#            if "input_ids" in inputs and "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
#                tmp_attention_mask = copy.deepcopy(inputs["attention_mask"])
#                tmp_attention_mask = tmp_attention_mask.bool()
#                tmp_attention_mask = tmp_attention_mask | (inputs["input_ids"] == IMAGE_TOKEN_INDEX)
#                tmp_labels = [
#                    cur_labels[cur_attention_mask]
#                    for cur_labels, cur_attention_mask in zip(tmp_labels, tmp_attention_mask)
#                ]
#                tmp_labels = torch.stack(tmp_labels)
#                for i in range(tmp_labels.shape[1]):
#                    if tmp_labels[0, i] == 78191:
#                        tcm_logger.debug(f"In compute_loss(): assistant token at position {i}")
#                        break
#                
#        if self.label_smoother is not None and "labels" in inputs:
#            labels = inputs.pop("labels")
#        else:
#            labels = None
#
#        outputs = model(**inputs)
#
#        # assert (isinstance(outputs, tuple) and len(outputs) == 2) or isinstance(outputs, CausalLMOutputWithPast), '@tcm: Expected: CausalLMOutputWithPast or tuple(loss, logits tensor)'
#        # if isinstance(outputs, tuple):
#        #     logits = outputs[1]
#        #     loss_val = outputs[0]
#        # else:
#        #     logits = outputs.logits
#        #     loss_val = outputs.loss
#        #     # outputs.hidden_states: None
#        #     # outputs.attentions: None
#            
#        # output_ids = logits.argmax(dim=-1)
#        # # output_ids: [torch.Size([1, 5361]), torch.int64, cuda:1]
#        # debug_tensor("In mm_trainer.py: output_ids", output_ids)
#        # assert len(output_ids) == len(inputs['input_ids']), 'Same batch size required'
#        # decoded_outputs = self.tokenizer.batch_decode(output_ids[..., :min(100, output_ids.shape[-1])], skip_special_tokens=True)
#        # tcm_logger.debug(f'In mm_trainer.py: decoded_outputs={decoded_outputs}')    
#        
#        # Save past state if it exists
#        # TODO: this needs to be fixed and made cleaner later.
#        if self.args.past_index >= 0:
#            self._past = outputs[self.args.past_index]
#
#        if labels is not None:
#            unwrapped_model = self.accelerator.unwrap_model(model)
#            if _is_peft_model(unwrapped_model):
#                model_name = unwrapped_model.base_model.model._get_name()
#            else:
#                model_name = unwrapped_model._get_name()
#            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                loss = self.label_smoother(outputs, labels, shift_labels=True)
#            else:
#                loss = self.label_smoother(outputs, labels)
#        else:
#            if isinstance(outputs, dict) and "loss" not in outputs:
#                raise ValueError(
#                    "The model did not return a loss from the inputs, only the following keys: "
#                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                )
#            # We don't use .loss here since the model may return tuples instead of ModelOutput.
#            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#
#        return (loss, outputs) if return_outputs else loss
#
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_masks = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            debug_tensor("After prediction_step: logits", logits)
            debug_tensor("After prediction_step: labels", labels)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            tcm_logger.debug(f"main_input_name: {main_input_name}")
            if isinstance(inputs[main_input_name], torch.Tensor):
                debug_tensor(f"In evaluation_loop(): inputs['{main_input_name}']", inputs[main_input_name])
            debug_tensor(f"In evaluation_loop(): inputs['attention_mask']", inputs['attention_mask'])
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None
            inputs_mask = self._prepare_input(inputs['attention_mask']) if args.include_inputs_for_metrics else None
            tcm_logger.debug(f"type(inputs_decode): {type(inputs_decode)}")
            if isinstance(inputs_decode, torch.Tensor):
                debug_tensor(f"In evaluation_loop(): inputs_decode", inputs_decode)

            # if is_torch_xla_available():
            #     xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if inputs_mask is not None:
                inputs_mask = self.accelerator.pad_across_processes(inputs_mask, dim=1, pad_index=0)
                inputs_mask = self.gather_function((inputs_mask))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_masks.add(inputs_mask)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                debug_tensor("Before accelerator.pad_across_processes: logits", logits)
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                debug_tensor("Before gather_function: logits", logits)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    debug_tensor("Add to all_preds: logits", logits)
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    debug_tensor("Add to all_preds: labels", labels)
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    if args.include_inputs_for_metrics:
                        metrics = self.compute_metrics(
                            EvalPredictionWithMask(predictions=logits, label_ids=labels, inputs=inputs, masks=all_masks),
                            compute_result=is_last_step,
                        )
                    else:
                        metrics = self.compute_metrics(
                            EvalPredictionWithMask(predictions=logits, label_ids=labels),
                            compute_result=is_last_step,
                        )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_masks.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_masks = all_masks.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPredictionWithMask(predictions=all_preds, label_ids=all_labels, inputs=all_inputs, masks=all_masks)
                )
            else:
                metrics = self.compute_metrics(EvalPredictionWithMask(predictions=all_preds, label_ids=all_labels))
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_xpu_available():
                torch.xpu.empty_cache()
            elif is_mlu_available():
                torch.mlu.empty_cache()
            elif is_npu_available():
                torch.npu.empty_cache()
            elif is_torch_version(">=", "2.0") and is_mps_available():
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            pass
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        torch.cuda.memory._dump_snapshot(datetime.now().strftime('snapshot_%d_%H%M%S.pickle'))

        return loss.detach() / self.args.gradient_accumulation_steps


#    def prediction_step(
#        self,
#        model: nn.Module,
#        inputs: Dict[str, Union[torch.Tensor, Any]],
#        prediction_loss_only: bool,
#        ignore_keys: Optional[List[str]] = None,
#    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
#
#        """
#        Perform an evaluation step on `model` using `inputs`.
#
#        Subclass and override to inject custom behavior.
#
#        Args:
#            model (`nn.Module`):
#                The model to evaluate.
#            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
#                The inputs and targets of the model.
#
#                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
#                argument `labels`. Check your model's documentation for all accepted arguments.
#            prediction_loss_only (`bool`):
#                Whether or not to return the loss only.
#            ignore_keys (`List[str]`, *optional*):
#                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
#                gathering predictions.
#
#        Return:
#            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
#            logits and labels (each being optional).
#        """
#        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
#        # For CLIP-like models capable of returning loss values.
#        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
#        # is `True` in `model.forward`.
#        return_loss = inputs.get("return_loss", None)
#        if return_loss is None:
#            return_loss = self.can_return_loss
#        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False
#
#        inputs = self._prepare_inputs(inputs)
#        if ignore_keys is None:
#            if hasattr(self.model, "config"):
#                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
#            else:
#                ignore_keys = []
#
#        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
#        if has_labels or loss_without_labels:
#            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
#            if len(labels) == 1:
#                labels = labels[0]
#        else:
#            labels = None
#
#        with torch.no_grad():
#            if False:
#                pass
#            else:
#                if has_labels or loss_without_labels:
#                    with self.compute_loss_context_manager():
#                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
#                    loss = loss.mean().detach()
#
#                    if isinstance(outputs, dict):
#                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
#                    else:
#                        logits = outputs[1:]
#                else:
#                    loss = None
#                    with self.compute_loss_context_manager():
#                        outputs = model(**inputs)
#                    if isinstance(outputs, dict):
#                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
#                    else:
#                        logits = outputs
#                    # TODO: this needs to be fixed and made cleaner later.
#                    if self.args.past_index >= 0:
#                        self._past = outputs[self.args.past_index - 1]
#
#        if prediction_loss_only:
#            return (loss, None, None)
#
#        logits = nested_detach(logits)
#        if len(logits) == 1:
#            logits = logits[0]
#        debug_tensor("In prediction_step: logits", logits)
#        debug_tensor("In prediction_step: labels", labels)
#        for i in range(labels.shape[1]):
#            if labels[0, i] == 78191:
#                tcm_logger.debug(f"Assistant token at position {i}")
#                break
#
#        return (loss, logits, labels)

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     """
    #     Perform a training step on a batch of inputs.

    #     Subclass and override to inject custom behavior.

    #     Args:
    #         model (`nn.Module`):
    #             The model to train.
    #         inputs (`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.

    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument `labels`. Check your model's documentation for all accepted arguments.

    #     Return:
    #         `torch.Tensor`: The tensor with training loss on this batch.
    #     """
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)
    #     # if is_sagemaker_mp_enabled():
    #     #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
    #     #     return loss_mb.reduce_mean().detach().to(self.args.device)

    #     with self.compute_loss_context_manager():
    #         loss = self.compute_loss(model, inputs)

    #     del inputs
    #     tcm_logger.info(f"args.torch_empty_cache_steps: {self.args.torch_empty_cache_steps}")
    #     tcm_logger.info(f"state.global_step: {self.state.global_step}")
    #     if (
    #         self.args.torch_empty_cache_steps is not None
    #         and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #     ):
    #         # if is_xpu_available():
    #         #     torch.xpu.empty_cache()
    #         # elif is_mlu_available():
    #         #     torch.mlu.empty_cache()
    #         # elif is_npu_available():
    #         #     torch.npu.empty_cache()
    #         # elif is_torch_version(">=", "2.0") and is_mps_available():
    #         #     torch.mps.empty_cache()
    #         # else:
    #         #     torch.cuda.empty_cache()
    #         torch.cuda.empty_cache()

    #     kwargs = {}

    #     # For LOMO optimizers you need to explicitly use the learnign rate
    #     if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         kwargs["learning_rate"] = self._get_learning_rate()

    #     if self.args.n_gpu > 1:
    #         loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #     if self.use_apex:
    #         with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #             scaled_loss.backward()
    #     else:
    #         self.accelerator.backward(loss, **kwargs)

    #     return loss.detach() / self.args.gradient_accumulation_steps
