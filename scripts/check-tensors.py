import html
import os.path
import tqdm
import torch
import safetensors.torch
import gradio as gr
from torch import Tensor
from modules import script_callbacks, shared
from modules import sd_models
from modules.ui import create_refresh_button
from modules.ui_components import FormRow
from typing import List


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant="compact"):
                with FormRow(elem_id="modelmerger_models"):
                    model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="check_tensors_model_name", label="Stable Diffusion checkpoint")
                    create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_checkpoint_A")

                all_deviations = gr.Checkbox(label="Print all deviations", value=True)
                model_check = gr.Button("Check tensors", elem_id="check_tensors_button", label="Check", variant='primary')

            with gr.Column(variant="compact"):
                checker_results = gr.Textbox(elem_id="check_tensors_results", show_label=False)

            model_check.click(
                fn=check_tensors,
                inputs=[
                    model_name,
                    all_deviations
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


def check_tensors(model, all_deviations):
    output = ""
    sum_dev = 0
    max_dev = 0
    min_dev = 0

    if model == "":
        return "Please choose a model"

    shared.state.begin()
    shared.state.job = "check_tensors"

    model_info = sd_models.checkpoints_list[model]
    shared.state.textinfo = f"Loading {model_info.filename}..."
    checkpoint = load_model(model_info.filename)
    output += f"{model}\n"

    if "cond_stage_model.transformer.text_model.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.text_model.embeddings.position_ids"]
    elif "cond_stage_model.transformer.embeddings.position_ids" in checkpoint:
        check_tensor = checkpoint["cond_stage_model.transformer.embeddings.position_ids"]
    else:
        return "Invalid checkpoint file"

    output += f"{check_tensor}\n"
    output += f"Type: {check_tensor.dtype}\n"
    for i in range(torch.numel(check_tensor)):
        tensor_value = check_tensor.data[0, i]
        value_error = tensor_value-i
        sum_dev += abs(value_error)
        if value_error > max_dev:
            max_dev = value_error
        if value_error < min_dev:
            min_dev = value_error
        if all_deviations:
            output += f"{i}: {tensor_value:.5f}  {value_error:.5f}\n"

    output += f"\nMax deviation: {(max_dev):.5f}\n"
    output += f"Min deviation: {(min_dev):.5f}\n"
    output += f"Mean deviation: {(sum_dev/77):.5f}\n"

    shared.state.end()
    return output


script_callbacks.on_ui_tabs(add_tab)
