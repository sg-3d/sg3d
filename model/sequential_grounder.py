import contextlib
import math

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from model.build import MODEL_REGISTRY, BaseModel
from modules.build import build_module_by_name
from modules.utils import get_mlp_head

logger = get_logger(__name__)

def maybe_autocast(model, dtype='bf16', enabled=True):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = model.device != torch.device('cpu')

    if dtype == 'bf16':
        dtype = torch.bfloat16
    elif dtype == 'fp16':
        dtype == torch.float16
    else:
        dtype == torch.float32

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)
    else:
        return contextlib.nullcontext()

def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore
    """
    return self

@MODEL_REGISTRY.register()
class SequentialGrounder(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        # LLM
        if 'vicuna' in cfg.model.llm.name.lower():
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(
                cfg.model.llm.cfg_path, truncation_side=cfg.model.llm.truncation_side
            )
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_model = LlamaForCausalLM.from_pretrained(cfg.model.llm.cfg_path, torch_dtype=torch.float16)
            self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        else:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.llm.cfg_path, truncation_side=cfg.model.llm.truncation_side
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(cfg.model.llm.cfg_path, torch_dtype=torch.float16)

        logger.info(f"Build {cfg.model.llm.name} from {cfg.model.llm.cfg_path}")

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        self.llm_model.train = disabled_train
        logger.info("Freeze LLM")

        # 2D vision
        self.img_encoder = build_module_by_name(cfg.model.vision2d)
        self.img_proj = nn.Linear(
            self.img_encoder.out_channels, self.llm_model.config.hidden_size
        )

        # 3D vision
        self.pcd_encoder = build_module_by_name(cfg.model.vision3d)
        self.pcd_proj = nn.Linear(
            cfg.model.vision3d.hidden_dim, self.llm_model.config.hidden_size
        )

        # type embedding
        # self.img_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)
        # self.pcd_type_embed = nn.Parameter(torch.zeros(self.llm_model.config.hidden_size), requires_grad=True)

        # LoRA
        if cfg.model.llm.lora.flag:
            logger.info(f"Apply LoRA with configs: {cfg.model.llm.lora}")
            lora_config = LoraConfig(
                r=cfg.model.llm.lora.rank,
                lora_alpha=cfg.model.llm.lora.alpha,
                target_modules=cfg.model.llm.lora.target_modules,
                lora_dropout=cfg.model.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)

        self.max_context_len = cfg.model.llm.max_context_len
        self.max_out_len = cfg.model.llm.max_out_len

        # additional text x multi-modal tokens fusion
        self.clip_txt_guidance = cfg.model.clip_txt_guidance.flag
        if self.clip_txt_guidance:
            logger.info("Add CLIP semantics guidance")
            self.clip_model = clip.load('RN50')[0]
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
            self.clip_model.train = disabled_train
            self.clip_proj = nn.Linear(cfg.clip_txt_guidance.clip_out_dim, self.llm_model.config.hidden_size)
        
        # grounding head
        self.ground_head = build_module_by_name(cfg.model.ground_head)
        self.obj_cls_head = get_mlp_head(4096, 768, 607, 0.3)
        self.pre_grounding = cfg.model.get('pre_grounding', True)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def count_params(self, parameters):
        tot = sum([math.prod(p.shape) for p in parameters])
        return tot

    def show_params_size(self, tot):
        if tot >= 1e9:
            return '{:.1f}B'.format(tot / 1e9)
        elif tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}k'.format(tot / 1e3)

    def get_opt_params(self):
        def get_lr(cfg, default_lr):
            return default_lr if cfg is None or cfg.get("lr") is None else cfg.get("lr")
        
        optimizer_grouped_parameters = []
        for name, module in self._modules.items():
            module_cfg = self.cfg.model.get(name)
            lr = get_lr(module_cfg, self.cfg.solver.lr)
            if lr != self.cfg.solver.lr:
                print(f"Change lr from default {self.cfg.solver.lr} to {lr} for {name} module.")
            optimizer_grouped_parameters += [{'params': [p for n, p in module.named_parameters() if p.requires_grad == True], 'name': name, 'lr': lr}]
        optimized_parameters = [p for group in optimizer_grouped_parameters for p in group['params']]
        print(f'actual train parameters: {self.show_params_size(self.count_params(optimized_parameters))}')    
        return optimizer_grouped_parameters
    
    def get_learnable_named_params(self):
        learnable_named_params = {}
        frozen_named_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                learnable_named_params.update({n: p})
            else:
                frozen_named_params.update({n: p})
        learnable_params_size = self.count_params(learnable_named_params.values())
        frozen_params_size = self.count_params(frozen_named_params.values())
        logger.info(
            f"Build sequential grounder with {self.show_params_size(learnable_params_size+frozen_params_size)} parameters, " +
            f"{self.show_params_size(learnable_params_size)} learnable and " +
            f"{self.show_params_size(frozen_params_size)} frozen"
        )
        logger.info(f"🧊 Frozen parameters: {list(frozen_named_params.keys())}")
        logger.info(f"🔥 Tuned parameters: {list(learnable_named_params.keys())}")
        
        return learnable_named_params

    def build_right_justified_sequence(self, data_dict):
        """
        Concat six sequences: `prompt_before_obj`, `prompt_middle_1`, `img_tokens`, `prompt_middle_2`, `obj_tokens`, `prompt_after_obj`.
        Return right justified sequence for causal LM: <pad>, <role/situation>, <img>, <objs>, <instruction>.
        """
        device = self.device
        bs = len(data_dict['prompt_before_obj'])

        self.llm_tokenizer.padding_side = 'left'
        text_input_tokens_pre = self.llm_tokenizer(
            data_dict['prompt_before_obj'],
            return_tensors='pt',
            padding='longest'
        ).to(device)   # [PAD, BOS, tokens], (B, T1)

        text_input_tokens_mid1 = self.llm_tokenizer(
            data_dict['prompt_middle_1'],
            return_tensors='pt',
            padding='longest'
        ).to(device)

        img_tokens = data_dict['img_tokens'].to(device)
        img_masks = data_dict['img_masks'].to(device)
        img_masks = img_masks.reshape(-1, 1).repeat(1, img_tokens.size(1))

        text_input_tokens_mid2 = self.llm_tokenizer(
            data_dict['prompt_middle_2'],
            return_tensors='pt',
            padding='longest'
        ).to(device)

        obj_tokens = data_dict['obj_tokens'].to(device)
        obj_masks = data_dict['obj_masks'].to(device)

        # additional clip fusion
        if self.clip_txt_guidance:
            with torch.no_grad():
                clip_fts = self.clip_model.encode_text(
                    clip.tokenize(data_dict['prompt_after_obj'], truncate=True).to(device)
                )
            clip_fts = self.clip_proj(clip_fts)
            # B, N, C
            img_tokens = torch.einsum('bnc,bc->bnc', img_tokens, clip_fts)
            obj_tokens = torch.einsum('bnc,bc->bnc', obj_tokens, clip_fts)

        self.llm_tokenizer.padding_side = 'right'   # no need to be 'left', as padding tokens will be shifted
        self.llm_tokenizer.truncation_side = 'left'   # truncate history
        text_input_tokens_post = self.llm_tokenizer(
            data_dict['prompt_after_obj'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
        ).to(device)   # [BOS, tokens, PAD], (B, T3)

        assert text_input_tokens_mid1.attention_mask.all() and text_input_tokens_mid2.attention_mask.all(), \
               "prompt_middle should be the same and thus no padding"

        # remove bos, make "tokenize subseq and concat" equivalent to "tokenize the whole seq"
        text_input_tokens_mid1.input_ids = text_input_tokens_mid1.input_ids[:, 1:]
        text_input_tokens_mid1.attention_mask = text_input_tokens_mid1.attention_mask[:, 1:]
        text_input_tokens_mid2.input_ids = text_input_tokens_mid2.input_ids[:, 1:]
        text_input_tokens_mid2.attention_mask = text_input_tokens_mid2.attention_mask[:, 1:]
        text_input_tokens_post.input_ids = text_input_tokens_post.input_ids[:, 1:]
        text_input_tokens_post.attention_mask = text_input_tokens_post.attention_mask[:, 1:]
        for i in range(bs):
            if not img_masks[i].any():
                # no image input, also mask the text prompt for image tokens
                text_input_tokens_mid1.attention_mask[i].fill_(0)

        inputs_embeds_pre = self.llm_model.get_input_embeddings()(text_input_tokens_pre.input_ids)
        inputs_embeds_mid1 = self.llm_model.get_input_embeddings()(text_input_tokens_mid1.input_ids)
        inputs_embeds_mid2 = self.llm_model.get_input_embeddings()(text_input_tokens_mid2.input_ids)
        inputs_embeds_post = self.llm_model.get_input_embeddings()(text_input_tokens_post.input_ids)

        # since img_tokens, prompt_mid, obj_tokens are fixed length without padding, we concat them first
        inputs_embeds_mid = torch.cat([inputs_embeds_mid1, img_tokens, inputs_embeds_mid2, obj_tokens], dim=1)
        attn_mask_mid = torch.cat(
            [text_input_tokens_mid1.attention_mask, img_masks, text_input_tokens_mid2.attention_mask, obj_masks],
            dim=1,
        )

        post_pad_length = torch.logical_not(text_input_tokens_post.attention_mask).sum(-1)

        bs, l1, hidden_dim = inputs_embeds_pre.shape
        _, l2, _ = inputs_embeds_mid.shape
        _, l3, _ = inputs_embeds_post.shape

        inputs_embeds = torch.zeros(bs, l1+l2+l3, hidden_dim).type(inputs_embeds_pre.dtype).to(device)
        attention_mask = torch.zeros(bs, l1+l2+l3).type(obj_masks.dtype).to(device)

        # assign by chunks
        for i in range(bs):
            post_pad_len = post_pad_length[i]

            if post_pad_len > 0:
                inputs_embeds[i, :post_pad_len] = inputs_embeds_post[i, -post_pad_len:]
                attention_mask[i, :post_pad_len] = 0
                inputs_embeds[i, post_pad_len+l1+l2:] = inputs_embeds_post[i, :-post_pad_len]
                attention_mask[i, post_pad_len+l1+l2:] = 1
            else:
                # no padding
                inputs_embeds[i, -l3:] = inputs_embeds_post[i]
                attention_mask[i, -l3:] = 1

            inputs_embeds[i, post_pad_len: post_pad_len+l1] = inputs_embeds_pre[i]
            attention_mask[i, post_pad_len: post_pad_len+l1] = text_input_tokens_pre.attention_mask[i]

            inputs_embeds[i, post_pad_len+l1: post_pad_len+l1+l2] = inputs_embeds_mid[i]
            attention_mask[i, post_pad_len+l1: post_pad_len+l1+l2] = attn_mask_mid[i]

        return inputs_embeds, attention_mask, (l1, l2, l3)

    def forward(self, data_dict):
        if not self.training and self.cfg.eval.get('predict_mode', False):
            return self.generate(data_dict=data_dict)
        """
        data_dict requires keys:
        # input
        prompt_before_obj: list of str, (B,)
        prompt_middle_1: list of str, (B,)
        prompt_middle_2: list of str, (B,)
        prompt_after_obj: list of str, (B,)
        obj_fts: (B, N, P, 6), xyz + rgb
        obj_masks: (B, N), 1 valid and 0 masked
        obj_locs: (B, N, 6), xyz + whd
        anchor_locs: (B, 3)
        anchor_orientation: (B, C)
        img_fts: (B, 3, H, W), rgb
        img_masks: (B, 1), 1 valid and 0 masked
        # output
        output_gt: list of str, (B,)
        """
        device = self.device
        bs = len(data_dict['prompt_after_obj'])
        data_dict['bs'] = bs
        if 'obj_tokens' not in data_dict:
            # obtain obj tokens
            data_dict = self.pcd_encoder(data_dict)
            # TO CHANGE FOR DEBUG
            #self.llm_model.float()
            #data_dict['obj_tokens'] = torch.zeros((data_dict['obj_locs'].shape[0], data_dict['obj_locs'].shape[1], 256)).to(device=device)

        data_dict['obj_tokens'] = self.pcd_proj(data_dict['obj_tokens'].to(device))
        # data_dict['obj_tokens'] = data_dict['obj_tokens'] + self.pcd_type_embed

        data_dict['img_tokens'] = self.img_proj(self.img_encoder(data_dict['img_fts']))
        # data_dict['img_tokens'] = data_dict['img_tokens'] + self.img_type_embed
        
        # build input embdes and record prompt position
        inputs_embeds, attention_mask, input_length = self.build_right_justified_sequence(data_dict=data_dict)
        obj_token_length = data_dict['obj_masks'].shape[1]
        # (B, T1+O+T2, D), (B, T1+O+T2)

        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in data_dict['output_gt']],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_out_len,
        ).to(device)
        # record position for special token [SOS]
        grd_token_id = self.llm_tokenizer.convert_tokens_to_ids(['<s>'])[0]
        out_input_ids_remove_first_sos = text_output_tokens.input_ids.clone()
        out_input_ids_remove_first_sos[:, 0] = -100
        grd_ind_0, grd_ind_1 = (out_input_ids_remove_first_sos == grd_token_id).nonzero(as_tuple=True)
        

        text_output_embeds = self.llm_model.get_input_embeddings()(text_output_tokens.input_ids)   # (B, T3, D)
        inputs_embeds = torch.cat([inputs_embeds, text_output_embeds], dim=1)   # (B, T1+O+T2+T3, D)
        attention_mask = torch.cat([attention_mask, text_output_tokens.attention_mask], dim=1)   # (B, T1+O+T2+T3)

        # construct targets
        targets = torch.zeros_like(attention_mask).long().fill_(-100)   # (B, T1+O+T2+T3)

        # only apply loss to answer tokens
        targets_idx = text_output_tokens.attention_mask.bool()
        targets[:, -targets_idx.shape[1]:][targets_idx] = text_output_tokens.input_ids[targets_idx]

        # do not predict bos token, regard it as condition instead
        targets[:, -targets_idx.shape[1]] = -100

        with maybe_autocast(self):
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        logits = outputs.logits.float()
        last_hidden_state = outputs.hidden_states[-1]

        # different from the loss inside `llm_model.forward`, here we take mean of each sequence instead of sum
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        num_tokens_for_loss = (shift_labels >= 0).int().sum(1)   # (B,)

        shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
        shift_labels = rearrange(shift_labels, 'b t -> (b t)')

        shift_labels = shift_labels.to(shift_logits.device)
        
        # record for llm loss
        data_dict['llm_logits'] = shift_logits
        data_dict['llm_labels'] = shift_labels
        data_dict['num_tokens_for_loss'] = num_tokens_for_loss
        
        # record for grounding loss
        grd_list = []
        obj_list = []
        mask_list = []
        for step in range(len(grd_ind_0)):
            batch_ind = grd_ind_0[step]
            grd_token_ind = grd_ind_1[step]
            if self.pre_grounding:
                output_obj_tokens = data_dict['obj_tokens'][batch_ind]
            else:
                output_obj_tokens = last_hidden_state[batch_ind, input_length[0] + input_length[1] - obj_token_length : input_length[0] + input_length[1], :]
            output_grd_tokens = last_hidden_state[batch_ind, sum(input_length) + grd_token_ind:sum(input_length) + grd_token_ind + 1, :]
            grd_list.append(output_grd_tokens)
            obj_list.append(output_obj_tokens)
            mask_list.append(data_dict['obj_masks'][batch_ind])
        output_obj = torch.stack(obj_list).float()
        output_grd = torch.stack(grd_list).float()
        data_dict['ground_logits'] = self.ground_head(output_obj, output_grd, torch.stack(mask_list))
        data_dict['ground_label'] = torch.concat(data_dict['tgt_object_id'], dim=0)
        
        # record for cls loss
        #obj_cls_post_embeds = last_hidden_state[:, input_length[0] + input_length[1] - obj_token_length : input_length[0] + input_length[1], :].float()
        obj_cls_post_embeds = data_dict['obj_tokens'].float()
        data_dict['obj_cls_post_logits'] = self.obj_cls_head(obj_cls_post_embeds)
        return data_dict

    @torch.no_grad()
    def generate(
        self,
        data_dict,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=3.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        """
        data_dict requires the same keys as forward() except output_gt
        """
        device = self.device
        bs = len(data_dict['prompt_after_obj'])
        data_dict['bs'] = bs
        if 'obj_tokens' not in data_dict:
            # obtain obj tokens
            data_dict = self.pcd_encoder(data_dict)
            # TO CHANGE FOR DEBUG
            #self.llm_model.float()
            #data_dict['obj_tokens'] = torch.zeros((data_dict['obj_locs'].shape[0], data_dict['obj_locs'].shape[1], 256)).to(device=device)

        data_dict['obj_tokens'] = self.pcd_proj(data_dict['obj_tokens'].to(device))
        # data_dict['obj_tokens'] = data_dict['obj_tokens'] + self.pcd_type_embed

        data_dict['img_tokens'] = self.img_proj(self.img_encoder(data_dict['img_fts']))
        # data_dict['img_tokens'] = data_dict['img_tokens'] + self.img_type_embed

        inputs_embeds, attention_mask, input_length = self.build_right_justified_sequence(data_dict=data_dict)
        obj_token_length = data_dict['obj_masks'].shape[1]
        
        # give bos token as condition
        bos_tokens = self.llm_tokenizer(
            [self.llm_tokenizer.bos_token] * bs,
            return_tensors='pt',
        ).to(device)
        bos_tokens_ids = bos_tokens.input_ids[:, 0:1]   # (B, 1)
        bos_tokens_attn = bos_tokens.attention_mask[:, 0:1]   # (B, 1)

        # prepare a `bos_token`
        bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        with maybe_autocast(self):
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True
            )
        # note output_ids_idx - 1 = step idx, because we do not preduct [BOS] 
        beam_indices = outputs.beam_indices # bs x step, beam indices range (bsxbeam)
        scores = outputs.scores # step x (bs x beam) x vocab
        hidden_states = outputs.hidden_states # step x layer x (bs x beam) x token_num x hidden_dim
        outputs = outputs.sequences # bs x output_ids
        outputs[outputs == self.llm_tokenizer.unk_token_id] = self.llm_tokenizer.eos_token_id
        # data_dict['output_tokens'] = outputs   # unable to gather variable-length tensors
        
        # record for grounding
        grd_token_id = self.llm_tokenizer.convert_tokens_to_ids(['<s>'])[0]
        out_input_ids_remove_first_sos = outputs.clone()
        out_input_ids_remove_first_sos[:, 0] = -100
        grd_ind_0, grd_ind_1 = (out_input_ids_remove_first_sos == grd_token_id).nonzero(as_tuple=True)
        
        grd_list = []
        grd_batch_ind_list = []
        obj_list = []
        mask_list = []
        if len(grd_ind_0) > 0:
            for step in range(len(grd_ind_0)):
                batch_ind = grd_ind_0[step]
                grd_token_ind = grd_ind_1[step]
                #output_obj_tokens = last_hidden_state[batch_ind, input_length[0] + input_length[1] - obj_token_length : input_length[0] + input_length[1], :]
                output_obj_tokens = data_dict['obj_tokens'][batch_ind]
                output_grd_tokens = hidden_states[grd_token_ind-1][-1][beam_indices[batch_ind, grd_token_ind-1]][-1].unsqueeze(0) # grd_token_ind - 1 because first token is sos
                grd_list.append(output_grd_tokens)
                grd_batch_ind_list.append(batch_ind)
                obj_list.append(output_obj_tokens)
                mask_list.append(data_dict['obj_masks'][batch_ind])
            output_obj = torch.stack(obj_list).float()
            output_grd = torch.stack(grd_list).float()
            data_dict['ground_logits'] = self.ground_head(output_obj, output_grd, torch.stack(mask_list)) 
        else:
            data_dict['ground_logits'] = None
        data_dict['ground_label'] = torch.concat(data_dict['tgt_object_id'], dim=0)
        data_dict['grd_batch_ind_list'] = grd_batch_ind_list
        
        output_txt = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_txt = [txt.strip() for txt in output_txt]
        data_dict['output_txt'] = output_txt
        return data_dict

    @torch.no_grad()
    def predict_answers(self, data_dict, answer_list, num_ans_candidates=128):
        """
        (1) Generate the first token and select most probable candidates (num_ans_candidates)
        (2) Then select answers from answer list, which start with the probable tokens
        (3) Lastly, use the selected answers as the ground-truth labels and calculate LM loss
        Return the answer that minimize the loss as the predicted answer
        """
        device = self.device
        num_ans_candidates = min(num_ans_candidates, len(answer_list))

        self.llm_tokenizer.padding_side = 'right'
        answer_candidates = self.llm_tokenizer(
            answer_list, padding='longest', return_tensors='pt'
        ).to(device)

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        # (1)
        if 'obj_tokens' not in data_dict:
            data_dict = self.pcd_encoder(data_dict)

        data_dict['obj_tokens'] = self.pcd_proj(data_dict['obj_tokens'].to(device))
        # data_dict['obj_tokens'] = data_dict['obj_tokens'] + self.pcd_type_embed

        data_dict['img_tokens'] = self.img_proj(self.img_encoder(data_dict['img_fts']))
        # data_dict['img_tokens'] = data_dict['img_tokens'] + self.img_type_embed

        inputs_embeds, attention_mask, input_length = self.build_right_justified_sequence(data_dict=data_dict)
        bs = inputs_embeds.shape[0]

        # give bos token as condition
        bos_tokens_ids = answer_ids[0, 0].view(1, 1).repeat(bs, 1)   # (B, 1)
        bos_tokens_attn = answer_atts[0, 0].view(1, 1).repeat(bs, 1)   # (B, 1)

        bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        with maybe_autocast(self):
            start_output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
        logits = start_output.logits[:, -1, :]   # first predicted token's logit

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)
        # (bs, num_ans_candidates)

        # (2)
        ans_ids = []
        ans_atts = []
        for topk_id in topk_ids:
            ans_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            ans_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        ans_ids = torch.cat(ans_ids, dim=0)
        ans_atts = torch.cat(ans_atts, dim=0)
        # (B * num_ans_candidates, T3)

        inputs_embeds = inputs_embeds.repeat_interleave(num_ans_candidates, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_ans_candidates, dim=0)
        # (B * num_ans_candidates, T1+O+T2+1, D), (B * num_ans_candidates, T1+O+T2+1)

        # truncate the appended bos token before concat
        inputs_embeds = inputs_embeds[:, :-1, :]
        attention_mask = attention_mask[:, :-1]
        # (B * num_ans_candidates, T1+O+T2, D), (B * num_ans_candidates, T1+O+T2)

        ans_embeds = self.llm_model.get_input_embeddings()(ans_ids)   # (B * num_ans_candidates, T3, D)
        inputs_embeds = torch.cat([inputs_embeds, ans_embeds], dim=1)   # (B * num_ans_candidates, T1+O+T2+T3, D)
        attention_mask = torch.cat([attention_mask, ans_atts], dim=1)   # (B * num_ans_candidates, T1+O+T2+T3)

        targets_ids = torch.zeros_like(attention_mask).long().fill_(-100)   # (B * num_ans_candidates, T1+O+T2+T3)
        # only apply loss to answer tokens
        targets_idx = ans_atts.bool()
        targets_ids[:, -targets_idx.shape[1]:][targets_idx] = ans_ids[targets_idx]

        # ignore the prediction of bos token
        targets_ids[:, -targets_idx.shape[1]] = -100

        # (3)
        with maybe_autocast(self):
            output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=targets_ids,
                return_dict=True
            )

        logits = output.logits.float()

        # get loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_ids[..., 1:].contiguous()
        num_tokens_for_loss = (shift_labels >= 0).int().sum(1)

        shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
        shift_labels = rearrange(shift_labels, 'b t -> (b t)')

        shift_labels = shift_labels.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')   # get loss per token

        loss = rearrange(loss, '(b t) -> b t', b = bs * num_ans_candidates)
        loss = loss.sum(1) / num_tokens_for_loss   # get loss per sequence, average over tokens
        loss = rearrange(loss, '(b1 b2) -> b1 b2', b1=bs)

        max_topk_ids = (-loss).argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        data_dict['answer_id'] = max_ids
        data_dict['output_txt'] = [answer_list[max_id] for max_id in max_ids]

        return data_dict
