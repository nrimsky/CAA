import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.helpers import add_vector_after_position, find_instruction_end_postion
from utils.tokenize_llama import tokenize_llama
from typing import Tuple, Optional


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False
        self.do_projection = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
                do_projection=self.do_projection,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False):
        self.add_activations = activations
        self.do_projection = do_projection

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []


class Llama7BChatWrapper:
    def __init__(self, token, system_prompt):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", use_auth_token=token
        ).to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_after_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.after_position = pos

    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        tokens = tokenize_llama(self.tokenizer, self.system_prompt, [(prompt, None)])
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate_text_with_conversation_history(
        self, history: Tuple[str, Optional[str]], max_new_tokens=50
    ) -> str:
        tokens = tokenize_llama(
            self.tokenizer, self.system_prompt, history, no_final_eos=True
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=50):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_after_positions(instr_pos)
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0]

    def get_logits(self, tokens):
        with t.no_grad():
            logits = self.model(tokens).logits
            return logits
        
    def get_logits_with_conversation_history(self, history: Tuple[str, Optional[str]]):
        tokens = tokenize_llama(
            self.tokenizer, self.system_prompt, history, no_final_eos=True
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, do_projection=False):
        self.model.model.layers[layer].add(activations, do_projection)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
