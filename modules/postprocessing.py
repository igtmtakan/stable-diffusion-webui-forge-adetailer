import os

from PIL import Image

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, infotext_utils, extras
from modules.shared import opts


def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    devices.torch_gc()

    shared.state.begin(job="extras")

    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = images.fix_image(img)
                    fn = ''
                else:
                    image = images.read(os.path.abspath(img.name))
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            assert input_dir, 'input directory not selected'

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, 'image not selected'
            yield image, None

    if extras_mode == 2 and output_dir != '':
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ''

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
                if image_data is None or not hasattr(image_data, 'mode'):
                    print(f"[Warning] Failed to load valid image from: {image_placeholder}")
                    continue
            except Exception as e:
                print(f"[Warning] Error reading image {image_placeholder}: {e}")
                continue
        else:
            image_data = image_placeholder

        # Additional safety check for image_data
        if image_data is None or not hasattr(image_data, 'mode'):
            print(f"[Warning] Invalid image data: {type(image_data)}")
            continue

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        # ADetailer処理（Extrasタブ用）
        adetailer_processed = False
        if len(args) >= 6:  # ADetailerパラメータが含まれている場合
            try:
                # 最後の6つの引数がADetailerパラメータ
                adetailer_args = args[-6:]
                if len(adetailer_args) == 6:
                    adetailer_enable, adetailer_model, adetailer_detection_model, adetailer_prompt_enhancement, adetailer_confidence, adetailer_mask_blur = adetailer_args
                    print(f"[Extras] ADetailer params: enable={adetailer_enable}, inpaint_model={adetailer_model}, detection_model={adetailer_detection_model}")

                    if adetailer_enable and hasattr(extras, 'run_adetailer_extras_wrapper'):
                        print(f"[Extras] Running ADetailer processing...")
                        processed_image, message = extras.run_adetailer_extras_wrapper(
                            initial_pp.image,
                            adetailer_enable,
                            adetailer_model,
                            adetailer_detection_model,
                            adetailer_prompt_enhancement,
                            adetailer_confidence,
                            adetailer_mask_blur
                        )
                        if processed_image is not None:
                            initial_pp.image = processed_image
                            infotext += f"\nADetailer: {message}"
                            adetailer_processed = True
                            print(f"[Extras] ADetailer processing completed: {message}")
                        else:
                            print(f"[Extras] ADetailer processing failed: {message}")
            except Exception as e:
                print(f"[Extras] ADetailer processing error: {e}")

        # 他のスクリプトを実行（ADetailerパラメータを除外）
        script_args = args[:-5] if len(args) >= 5 else args
        scripts.scripts_postproc.run(initial_pp, script_args)

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ''
                forced_filename = None

            infotext = ", ".join([k if k == v else f'{k}: {infotext_utils.quote(v)}' for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

                if pp.caption:
                    caption_filename = os.path.splitext(fullfn)[0] + ".txt"
                    existing_caption = ""
                    try:
                        with open(caption_filename, encoding="utf8") as file:
                            existing_caption = file.read().strip()
                    except FileNotFoundError:
                        pass

                    action = shared.opts.postprocessing_existing_caption_action
                    if action == 'Prepend' and existing_caption:
                        caption = f"{existing_caption} {pp.caption}"
                    elif action == 'Append' and existing_caption:
                        caption = f"{pp.caption} {existing_caption}"
                    elif action == 'Keep' and existing_caption:
                        caption = existing_caption
                    else:
                        caption = pp.caption

                    caption = caption.strip()
                    if caption:
                        with open(caption_filename, "w", encoding="utf8") as file:
                            file.write(caption)

            # デバッグ情報を追加
            print(f"[Extras] Debug: extras_mode={extras_mode}, show_extras_results={show_extras_results}")
            print(f"[Extras] Debug: Condition result: {extras_mode != 2 or show_extras_results}")

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)
                print(f"[Extras] Debug: Added image to outputs. Total outputs: {len(outputs)}")
            else:
                print(f"[Extras] Debug: Skipped adding image to outputs due to condition")

    devices.torch_gc()
    shared.state.end()
    return outputs, ui_common.plaintext_to_html(infotext), ''


def run_postprocessing_webui(id_task, *args, **kwargs):
    return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    """old handler for API"""

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_enabled": True,
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "max_side_length": max_side_length,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "enable": True,
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "enable": True,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
