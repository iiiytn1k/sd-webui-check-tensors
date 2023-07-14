import html
import os.path
import tqdm
import torch
import safetensors.torch
import gradio as gr
from torch import Tensor
from modules import script_callbacks, shared
from modules import sd_models, hashes
from modules.ui import create_refresh_button
from modules.ui_components import FormRow
from typing import List


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="compact"):
                with FormRow():
                    model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="check_tensors_model_name", label="Stable Diffusion checkpoint")
                    create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")
                with FormRow():
                    calc_hashes = gr.Checkbox(label="Calculate hash", value=True)
                model_check = gr.Button("Check tensors", elem_id="check_tensors_button", label="Check", variant='primary')

            with gr.Column(variant="compact"):
                checker_results = gr.Textbox(elem_id="check_tensors_results", show_label=False)

            model_check.click(
                fn=check_tensors,
                inputs=[
                    model_name,
                    calc_hashes
                ],
                outputs=[checker_results]
            )
    return [(ui, "CLIP tensors checker", "check_tensors")]


def load_model(path):
    if path.endswith(".safetensors"):
        model_file = safetensors.torch.load_file(path, device="cpu")
    else:
        model_file = torch.load(path, map_location="cpu")

    state_dict = model_file["state_dict"] if "state_dict" in model_file else model_file
    return state_dict


def check_tensors(model, calc_hashes):
    output = ""
    wrong_index = []
    if model == "":
        return "Please choose a checkpoint"

    shared.state.begin()
    shared.state.job = "check_tensors"
    model_info = sd_models.checkpoints_list[model]
    output += f"{model_info.name}\n"

    if calc_hashes:
        #  Check for hash entry in cache.json. If not, then calculate.
        sha256_value = hashes.sha256(model_info.filename, "checkpoint/" + model_info.name)
        output += f"Hashes:\n"
        output += f"AUTOV2: {sha256_value[0:10]}\n"
        output += f"AUTOV1: {model_info.hash}\n\n"

    shared.state.textinfo = f"Loading {model_info.filename}..."
    checkpoint = load_model(model_info.filename)

    if "cond_stage_model.transformer.text_model.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    elif "cond_stage_model.transformer.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.embeddings.position_ids"]
    else:
        output += f"Invalid checkpoint file or checkpoint in SDv2/SDXL format"
        return output

    output += f"{check_tensor}\n"
    output += f"Type: {check_tensor.dtype}\n"
    for i in range(torch.numel(check_tensor)):
        tensor_value = check_tensor.data[0, i]
        value_error = tensor_value-i
        if abs(value_error)>0.0001:
            wrong_index.append(i)
    if len(wrong_index)>0:
        output += f"\nWrong CLIP indexes: {wrong_index}\n"
        output += f"It is recommended to fix this checkpoint.\n"
    shared.state.end()
    return output


script_callbacks.on_ui_tabs(add_tab)
