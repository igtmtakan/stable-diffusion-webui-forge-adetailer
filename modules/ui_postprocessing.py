import gradio as gr
from modules import scripts, shared, ui_common, postprocessing, call_queue, ui_toprow, extras
import modules.infotext_utils as parameters_copypaste
from modules.ui_components import ResizeHandleRow
from modules_forge.forge_canvas.canvas import ForgeCanvas


def create_ui():
    dummy_component = gr.Textbox(visible=False)
    tab_index = gr.State(value=0)

    with ResizeHandleRow(equal_height=False, variant='compact'):
        with gr.Column(variant='compact'):
            with gr.Tabs(elem_id="mode_extras"):
                with gr.TabItem('Single Image', id="single_image", elem_id="extras_single_tab") as tab_single:
                    extras_image = gr.Image(label="Source", interactive=True, type="pil", elem_id="extras_image", image_mode="RGBA", height="55vh")

                with gr.TabItem('Batch Process', id="batch_process", elem_id="extras_batch_process_tab") as tab_batch:
                    image_batch = gr.Files(label="Batch Process", interactive=True, elem_id="extras_image_batch")

                with gr.TabItem('Batch from Directory', id="batch_from_directory", elem_id="extras_batch_directory_tab") as tab_batch_dir:
                    extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                    extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                    show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")

            script_inputs = scripts.scripts_postproc.setup_ui()

            # ADetailer section for Extras tab
            with gr.Accordion("ADetailer", open=False, elem_id="extras_adetailer_accordion"):
                adetailer_enable = gr.Checkbox(label="Enable ADetailer", value=False, elem_id="extras_adetailer_enable")

                with gr.Row():
                    # Inpaintモデルリストを動的に取得
                    try:
                        if hasattr(extras, 'get_adetailer_models_wrapper'):
                            inpaint_model_choices = extras.get_adetailer_models_wrapper()
                            print(f"[UI] ADetailer inpaint models loaded: {inpaint_model_choices}")
                        else:
                            inpaint_model_choices = ["None"]
                            print("[UI] get_adetailer_models_wrapper not found")
                    except Exception as e:
                        print(f"[UI] Failed to get ADetailer inpaint models: {e}")
                        inpaint_model_choices = ["None"]

                    # 顔検出モデルリストを動的に取得
                    try:
                        if hasattr(extras, 'get_detection_models_wrapper'):
                            detection_model_choices = extras.get_detection_models_wrapper()
                            print(f"[UI] ADetailer detection models loaded: {detection_model_choices}")
                        else:
                            detection_model_choices = ["None"]
                            print("[UI] get_detection_models_wrapper not found")
                    except Exception as e:
                        print(f"[UI] Failed to get ADetailer detection models: {e}")
                        detection_model_choices = ["None"]

                    print(f"[UI] Final inpaint model choices: {inpaint_model_choices}")
                    print(f"[UI] Final detection model choices: {detection_model_choices}")

                    adetailer_model = gr.Dropdown(
                        label="Inpaint Model",
                        choices=inpaint_model_choices,
                        value="None",
                        elem_id="extras_adetailer_model"
                    )
                    adetailer_detection_model = gr.Dropdown(
                        label="Detection Model",
                        choices=detection_model_choices,
                        value="None",
                        elem_id="extras_adetailer_detection_model"
                    )

                with gr.Row():
                    adetailer_prompt_enhancement = gr.Checkbox(
                        label="Auto Prompt Enhancement",
                        value=True,
                        elem_id="extras_adetailer_prompt_enhancement"
                    )

                with gr.Row():
                    adetailer_confidence = gr.Slider(
                        label="Detection Confidence",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.01,
                        elem_id="extras_adetailer_confidence"
                    )
                    adetailer_mask_blur = gr.Slider(
                        label="Mask Blur",
                        minimum=0,
                        maximum=64,
                        value=4,
                        step=1,
                        elem_id="extras_adetailer_mask_blur"
                    )

        with gr.Column():
            toprow = ui_toprow.Toprow(is_compact=True, is_img2img=False, id_part="extras")
            toprow.create_inline_toprow_image()
            submit = toprow.submit

            output_panel = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)

    tab_single.select(fn=lambda: 0, inputs=[], outputs=[tab_index])
    tab_batch.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
    tab_batch_dir.select(fn=lambda: 2, inputs=[], outputs=[tab_index])

    submit_click_inputs = [
        dummy_component,
        tab_index,
        extras_image,
        image_batch,
        extras_batch_input_dir,
        extras_batch_output_dir,
        show_extras_results,
        *script_inputs,
        adetailer_enable,
        adetailer_model,
        adetailer_detection_model,
        adetailer_prompt_enhancement,
        adetailer_confidence,
        adetailer_mask_blur
    ]

    submit.click(
        fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing_webui, extra_outputs=[None, '']),
        _js=f"submit_extras",
        inputs=submit_click_inputs,
        outputs=[
            output_panel.gallery,
            output_panel.generation_info,
            output_panel.html_log,
        ],
        show_progress=False,
    )

    parameters_copypaste.add_paste_fields("extras", extras_image, None)

    extras_image.change(
        fn=scripts.scripts_postproc.image_changed,
        inputs=[], outputs=[]
    )
