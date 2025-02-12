#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerateOutput

from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging as lgging
import logging
import sys
sys.path.append('.')
from longvu.resource_logging import *

from ..cambrian_arch import CambrianMetaForCausalLM, CambrianMetaModel
from ..mm_utils import extract_engagement_label

IS_XLA_AVAILABLE = False

logger = lgging.get_logger(__name__)
tcm_logger = logging.getLogger("tcm_logger")

class CambrianConfig(LlamaConfig):
    model_type = "cambrian_llama"

    debug = "debug"


class CambrianLlamaModel(CambrianMetaModel, LlamaModel):
    config_class = CambrianConfig

    def __init__(self, config: LlamaConfig):
        super(CambrianLlamaModel, self).__init__(config)

    def forward(
        self,
        # pyre-fixme[9]: input_ids has type `LongTensor`; used as `None`.
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `config`.
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute
        #  `gradient_checkpointing`.
        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `training`.
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                # pyre-fixme[9]: past_key_values has type
                #  `Optional[List[FloatTensor]]`; used as `DynamicCache`.
                # pyre-fixme[6]: For 1st argument expected
                #  `Optional[Tuple[Tuple[FloatTensor]]]` but got
                #  `Optional[List[FloatTensor]]`.
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # pyre-fixme[16]: `Optional` has no attribute `get_usable_length`.
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            # pyre-fixme[16]: `Optional` has no attribute `device`.
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `embed_tokens`.
            inputs_embeds = self.embed_tokens(input_ids)

        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute
        #  `_use_flash_attention_2`.
        self._use_flash_attention_2 = getattr(self, "_use_flash_attention_2", False)
        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `_use_sdpa`.
        self._use_sdpa = getattr(self, "_use_sdpa", True)
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds
        # hidden_states: [torch.Size([1, 5361, 3072]), torch.float32, cuda:1]
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `layers`.
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # pyre-fixme[16]: `CambrianLlamaModel` has no attribute
                #  `_gradient_checkpointing_func`.
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # pyre-fixme[16]: `CambrianLlamaModel` has no attribute `norm`.
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                # pyre-fixme[61]: `use_legacy_cache` is undefined, or not always
                #  defined.
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CambrianLlamaForCausalLM(LlamaForCausalLM, CambrianMetaForCausalLM):
    config_class = CambrianConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = CambrianLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        # pyre-fixme[9]: input_ids has type `LongTensor`; used as `None`.
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        final_vision_feature_size = None
        if isinstance(images, torch.Tensor):
            debug_tensor("images", images)
        elif isinstance(images, list) or isinstance(images, tuple):
            for i, image in enumerate(images):
                if isinstance(image, torch.Tensor):
                    debug_tensor(f"images_{i}", image)

        orig_labels = labels
        if inputs_embeds is None:
            with MeasureResourceUsage("CambrianLlamaForCausalLM -> forward -> prepare_inputs_labels_for_multimodal"):
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    multimodal_mask,
                    labels,
                    vision_tower_aux_feature_list,
                    vision_tower_aux_attention_masks_list,
                    final_vision_feature_size,
                    global_context_feature,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_aux_attention_masks_list,
                    image_sizes,
                )
                # inputs_embeds: [torch.Size([1, 5361, 3072]), torch.float32, cuda:1] for 61 frames
        
        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            # self.model.gradient_checkpointing = False

            # pyre-fixme[21]: Could not find module `torch_xla.utils.checkpoint`.
            from torch_xla.utils.checkpoint import checkpoint

            # self.model.gradient_checkpointing = True
            # pyre-fixme[16]: `CambrianLlamaModel` has no attribute
            #  `_gradient_checkpointing_func`.
            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = (
            output_attentions
            if output_attentions is not None
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute `config`.
            else self.config.output_attentions
        ) # False
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        ) # False
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        ) # True
        
        
        # training
        if IS_XLA_AVAILABLE:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            # pyre-fixme[29]: `CambrianLlamaModel` is not a function.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # pyre-fixme[61]: `vision_tower_aux_feature_list` is undefined, or
                #  not always defined.
                vision_tower_aux_feature_list=vision_tower_aux_feature_list,
                # pyre-fixme[61]: `vision_tower_aux_attention_masks_list` is
                #  undefined, or not always defined.
                vision_tower_aux_attention_masks_list=vision_tower_aux_attention_masks_list,
                final_vision_feature_size=final_vision_feature_size,
                # pyre-fixme[61]: `global_context_feature` is undefined, or not
                #  always defined.
                global_context_feature=global_context_feature,
            )

        # inference
        else:
            with MeasureResourceUsage("CambrianLlamaForCausalLM -> forward -> model.forward"):
                if hasattr(self, "vision_tower_aux_feature_list"):
                    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                    # pyre-fixme[29]: `CambrianLlamaModel` is not a function.
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        vision_tower_aux_feature_list=(
                            # pyre-fixme[61]: `vision_tower_aux_feature_list` is
                            #  undefined, or not always defined.
                            vision_tower_aux_feature_list
                            if inputs_embeds is None
                            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no
                            #  attribute `vision_tower_aux_feature_list`.
                            else self.vision_tower_aux_feature_list
                        ),
                        vision_tower_aux_attention_masks_list=(
                            # pyre-fixme[61]: `vision_tower_aux_attention_masks_list` is
                            #  undefined, or not always defined.
                            vision_tower_aux_attention_masks_list
                            if inputs_embeds is None
                            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no
                            #  attribute `vision_tower_aux_attention_masks_list`.
                            else self.vision_tower_aux_attention_masks_list
                        ),
                        final_vision_feature_size=(
                            final_vision_feature_size
                            if inputs_embeds is None
                            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no
                            #  attribute `final_vision_feature_size`.
                            else self.final_vision_feature_size
                        ),
                        global_context_feature=(
                            # pyre-fixme[61]: `global_context_feature` is undefined, or
                            #  not always defined.
                            global_context_feature
                            if inputs_embeds is None
                            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no
                            #  attribute `global_context_feature`.
                            else self.global_context_feature
                        ),
                    )
                else:
                    # pyre-fixme[29]: `CambrianLlamaModel` is not a function.
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions, # False
                        output_hidden_states=output_hidden_states, # False
                        return_dict=return_dict, # True
                        # final_vision_feature_size=final_vision_feature_size,
                    )
        # outputs: BaseModelOutputWithPast(last_hidden_state=tensor([[[-1.4433,  0.6050, -0.7025,  ..., 
        # # -0.2153,  2.8000,  0.6008]]], device='cuda:1', grad_fn=<MulBackward0>), past_key_values=None, 
        # hidden_states=None, attentions=None)
        with MeasureResourceUsage("CambrianLlamaForCausalLM -> forward -> lm_head, logits"):
            hidden_states = outputs[0]
            # hidden_states: [torch.Size([1, 5361, 3072]), torch.float32, cuda:0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(
                    self.vocab_size // self.config.pretraining_tp, dim=0
                )
                logits = [
                    F.linear(hidden_states, lm_head_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()
            # logits: [torch.Size([1, 5361, 128256]), torch.float32, cuda:0]

            loss = None
            if labels is not None:
                # Print out decoded assistant tokens to check during finetuning
                assert isinstance(labels, torch.Tensor), "@tcm: labels should be a tensor"
                num_toks = labels.shape[-1]
                assistant_logits = None
                for i in range(num_toks - 1, 0, -1):
                    if labels[..., i] == 128007:
                        tcm_logger.debug(f"In CambrianLlamaForCausalLM.forward(): Found assistant token at index {i-1}, cut from {i+1}")
                        assistant_logits = logits[..., i+1:, :]
                        break
                assert assistant_logits is not None and isinstance(assistant_logits, torch.Tensor), "@tcm: assistant_logits should be a tensor"
                assistant_outputs = assistant_logits.argmax(dim = -1)
                decoded_outputs = self.tokenizer.batch_decode(assistant_outputs, skip_special_tokens=True)
                tcm_logger.info(f"In CambrianLlamaForCausalLM.forward(): Decoded assistant outputs: {decoded_outputs}")
                pred = extract_engagement_label(decoded_outputs[0])
                if pred == -1:
                    tcm_logger.info(f"ENGAGEMENT LABEL UNKNOWN")

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                # shift_logits: [torch.Size([1, 5360, 128256]), torch.float32, cuda:0]
                shift_labels = labels[..., 1:].contiguous()
                # shift_labels: [torch.Size([1, 5360]), torch.int64, cuda:0]
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        for i in range(multimodal_mask.shape[0]):
            true_pos = []
            for j in range(multimodal_mask.shape[1]):
                if multimodal_mask[i, j] == True:
                    true_pos.append(j)
            # check if true_pos is a sequence of consecutive integers
            if len(true_pos) > 0 and true_pos == list(range(true_pos[0], true_pos[-1] + 1)):
                tcm_logger.debug(f"sample {i}: correct range [{true_pos[0]}, {true_pos[-1]}]")
            else:
                tcm_logger.debug(f"sample {i}: incorrect range {true_pos}")
        orig_logits = logits[multimodal_mask].view(logits.size(0), -1, logits.size(2))
        debug_tensor("In CambrianLlamaForCausalLM.forward(): orig_logits", orig_logits)
        debug_tensor("In CambrianLlamaForCausalLM.forward(): orig_labels", orig_labels)
        # assert orig_logits.shape[0:2] == orig_labels.shape[0:2], f"shape mismatch"
        for i in range(orig_labels.shape[0]):
            out_range = (-1, -1)
            for j in range(orig_labels.shape[1]):
                if orig_labels[i, j] == 78191:
                    out_range[0] = j + 1
                    break
            for j in range(out_range[0], orig_logits.shape[1]):
                if orig_labels[i, j] == 128009:
                    out_range[1] = j
                    break
            outs = orig_logits[i, out_range[0]:out_range[1], :].argmax(dim = -1).unsqueeze(0)
            decoded_outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            tcm_logger.info(f"sample {i}: decoded outputs: {decoded_outs}")

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (orig_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=logits,
            logits=orig_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                vision_tower_aux_feature_list,
                vision_tower_aux_attention_masks_list,
                final_vision_feature_size,
                global_context_feature,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
            #  `vision_tower_aux_feature_list`.
            self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
            #  `vision_tower_aux_attention_masks_list`.
            self.vision_tower_aux_attention_masks_list = (
                vision_tower_aux_attention_masks_list
            )
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
            #  `final_vision_feature_size`.
            self.final_vision_feature_size = final_vision_feature_size
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
            #  `global_context_feature`.
            self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # pyre-fixme[16]: `LlamaForCausalLM` has no attribute `generate`.
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("cambrian_llama", CambrianConfig)
AutoModelForCausalLM.register(CambrianConfig, CambrianLlamaForCausalLM)
