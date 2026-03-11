import os
import time
import json
from pathlib import Path
from functools import partial
import gradio as gr
from modules import script_callbacks, shared
from scripts.florence2 import Florence2


def sync_value(value):
    """同步所有标签页的选择器值"""
    return [value, value]


# send to txt2img or img2img
def get_send_js_code(target_tab_type):
    """使用指定的 ID: txt2img 或 img2img"""
    return f"""
        async function() {{
            function getTagText() {{
            let tagsBox = gradioApp().getElementById("wd14_tags_output_single");
            if (!tagsBox) return "";
            let ta = tagsBox.querySelector("textarea, input");
            if (ta && ta.value) return ta.value;
            return tagsBox.textContent.trim();
        }}

        // 图片 as base64
        async function getImageBase64FromImg(src) {{
            return new Promise((resolve) => {{
                let img = new window.Image();
                img.crossOrigin = "Anonymous";
                img.onload = function() {{
                    try {{
                        let canvas = document.createElement('canvas');
                        canvas.width = img.width;
                        canvas.height = img.height;
                        let ctx = canvas.getContext('2d');
                        ctx.drawImage(img, 0, 0);
                        let b64 = canvas.toDataURL('image/png').split(',')[1];
                        resolve(b64);
                    }} catch(e) {{
                        resolve("");
                    }}
                }};
                img.onerror = function() {{ resolve(""); }};
                img.src = src;
            }});
        }}
        // 调用
        let tags = getTagText();
        let image_b64 = "";
        let parent = gradioApp().getElementById("wd14_single_input");
        if (parent) {{
            let img = parent.querySelector("img");
            if (img && img.src) {{
                image_b64 = await getImageBase64FromImg(img.src);
            }}
        }}

        data = {{ tags, image_b64 }};
        console.log(data);

        var targetTabContentId = 'tab_{target_tab_type}'; // 使用您指定的 ID: tab_txt2img 或 tab_img2img
        var tabContent = gradioApp().getElementById(targetTabContentId);
        if (!tabContent) {{
            console.error("WD14 Tagger: 找不到目標分頁 ID: " + targetTabContentId);
            return [];
        }}
        // 尋找該 ID 區塊內的第一個 textarea
        var prompt_textarea = tabContent.querySelector('textarea');
        if (prompt_textarea) {{
            prompt_textarea.value = data.tags;
            prompt_textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            prompt_textarea.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }} else {{
            console.error("WD14 Tagger: 在 " + targetTabContentId + " 中找不到 textarea");
        }}
        // 跳转页面
        var tabsContainer = gradioApp().getElementById('tabs');
        if (!tabsContainer) tabsContainer = gradioApp().querySelector('.tabs');
        if (tabsContainer) {{
            var allTabDivs = Array.from(tabsContainer.children).filter(
                node => node.tagName === 'DIV' && node.classList.contains('tabitem')
            );
            // 找出目標 ID 在這些分頁中的索引 (Index)
            var targetIndex = -1;
            for (var i = 0; i < allTabDivs.length; i++) {{
                if (allTabDivs[i].id === targetTabContentId) {{
                    targetIndex = i;
                    break;
                }}
            }}
            if (targetIndex !== -1) {{
                var nav = tabsContainer.querySelector('.tab-nav');
                if (nav) {{
                    var buttons = nav.querySelectorAll('button');
                    if (buttons && buttons[targetIndex]) {{
                        buttons[targetIndex].click();
                    }} else {{
                        console.error("WD14 Tagger: 找不到對應索引的按鈕");
                    }}
                }} else {{
                    console.error("WD14 Tagger: 找不到 .tab-nav");
                }}
            }} else {{
                console.error("WD14 Tagger: 無法計算分頁索引，找不到 ID " + targetTabContentId + " 在 tabs 中的位置");
            }}
        }}
        return [];
    }}
    """


# def save_tags_to_txt(tags_dict):
#     """將标签保存到 TXT 文件"""
#     tags_dict = eval(tags_dict)
#     if not tags_dict:
#         return
#     for name, tags in tags_dict.items():
#         filename = WD14_DIR / f"{name}.txt"
#         with open(filename, "w") as f:
#             f.write(tags)
#     gr.Info("The label has been successfully saved")


def save_image(image, img_type="output"):
    """
    保存修改的图片
    :param image: PIL Image对象（Gradio传入的图片）
    :param img_type: 图片类型（output=输出图，mask=掩码图）
    :return: Gradio提示信息
    """
    if image is None:
        return gr.Info("No images available to save")

    file_name = f"florence2_{img_type}_{int(time.time())}.png"
    save_path = FLORENCE2_DIR / file_name

    try:
        image.save(save_path, format="PNG")
        return gr.Info(f"success to save：{save_path}")
    except Exception as e:
        return gr.Error(f"fail to save：{str(e)}")


def on_ui_tabs():
    def sync_opts_to_components():
        return [
            # florence2
            gr.update(value=shared.opts.florence2_model_text),
            gr.update(value=shared.opts.florence2_model_image),
            gr.update(value=shared.opts.florence2_Lora_text),
            gr.update(value=shared.opts.florence2_Lora_image),
            gr.update(value=shared.opts.florence2_task_text),
            gr.update(value=shared.opts.florence2_task_image),
            gr.update(value=shared.opts.florence2_num_beams_text),
            gr.update(value=shared.opts.florence2_num_beams_image),
            gr.update(value=shared.opts.florence2_max_token_text),
            gr.update(value=shared.opts.florence2_max_token_image),
            gr.update(value=shared.opts.florence2_dtype_text),
            gr.update(value=shared.opts.florence2_dtype_image),
            gr.update(value=shared.opts.florence2_attention_text),
            gr.update(value=shared.opts.florence2_attention_image),
            gr.update(value=shared.opts.florence2_keep_model_loaded_text),
            gr.update(value=shared.opts.florence2_keep_model_loaded_image),
            gr.update(value=shared.opts.florence2_show_json_text),
            gr.update(value=shared.opts.florence2_show_json_image),
            gr.update(value=shared.opts.florence2_fill_mask),
        ]

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        # florence2
        with gr.Tabs(elem_id="tabs"):
            with gr.Tab(label=I18N["Text Output"][default_lang], elem_id="text_output"):
                with gr.Row():
                    # input
                    with gr.Column(variant="panel"):
                        input_image_florence2_1_1 = gr.Image(
                            label=I18N["Input Image"][default_lang],
                            sources=["upload", "webcam"],
                            type="pil",
                            height=520,
                            object_fit="contain",
                            elem_id="florence2_input_image1_1",
                        )
                        with gr.Row():
                            florence2_model_selector1_1 = gr.Dropdown(
                                label=I18N["Select Model"][default_lang],
                                choices=florence2_backend.model_list,
                                value=shared.opts.florence2_model,
                                elem_id="florence2_model_selector1_1",
                            )
                            florence2_lora_selector1_1 = gr.Dropdown(
                                label=I18N["Select Lora"][default_lang],
                                choices=florence2_backend.lora_list,
                                value=shared.opts.florence2_Lora,
                                elem_id="florence2_lora_selector1_1",
                            )
                        florence2_task1_1 = gr.Dropdown(
                            label=I18N["Select Task"][default_lang],
                            choices=["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"],
                            value=shared.opts.florence2_task,
                            elem_id="florence2_task_selector1_1",
                        )
                        florence2_text_input1_1 = gr.Textbox(
                            label=I18N["Text Input (for specific tasks)"][default_lang],
                            lines=2,
                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                            elem_id="florence2_text_input1_1",
                        )
                        with gr.Row():
                            florence2_num_beams1_1 = gr.Number(
                                label=I18N["Num Beams"][default_lang],
                                value=shared.opts.florence2_num_beams,
                                minimum=1,
                                maximum=10,
                                elem_id="florence2_num_beams_selector1_1",
                            )
                            florence2_max_token1_1 = gr.Number(
                                label=I18N["Max token"][default_lang],
                                value=shared.opts.florence2_max_token,
                                minimum=1,
                                maximum=4096,
                                elem_id="florence2_max_token_selector1_1",
                            )
                        with gr.Row():
                            florence2_dtype1_1 = gr.Dropdown(
                                label=I18N["Select Precision"][default_lang],
                                choices=florence2_backend.dtype,
                                value=shared.opts.florence2_dtype,
                                elem_id="florence2_dtype_selector1_1",
                            )
                            florence2_attention1_1 = gr.Dropdown(
                                label=I18N["Select Attention Implementation"][default_lang],
                                choices=florence2_backend.attention_list,
                                value=shared.opts.florence2_attention,
                                elem_id="florence2_attention_selector1_1",
                            )
                        with gr.Row():
                            florence2_keep_model_loaded1_1 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector1_1")
                            florence2_show_json1_1 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_text, elem_id="florence2_is_show_json1_1")
                        with gr.Row():
                            florence2_interrogate_btn1_1 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn1_1")
                            florence2_unload_btn1_1 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn1_1")
                    # output
                    with gr.Column(variant="panel"):
                        florence2_tags_output1_1 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags1_1")
                        with gr.Row():
                            florence2_send_to_txt2img1 = gr.Button(I18N["Send to Txt2Img"][default_lang], elem_id="florence2_send_txt2img_btn1")
                            florence2_send_to_img2img1 = gr.Button(I18N["Send to Img2Img"][default_lang], elem_id="florence2_send_img2img_btn1")

                # 事件绑定
                florence2_interrogate_btn1_1.click(
                    fn=partial(florence2_backend.predict, output_type="text"),
                    inputs=[input_image_florence2_1_1, florence2_model_selector1_1, florence2_lora_selector1_1, florence2_task1_1, florence2_text_input1_1, florence2_num_beams1_1, florence2_max_token1_1, florence2_dtype1_1, florence2_attention1_1, florence2_keep_model_loaded1_1, florence2_show_json1_1],
                    outputs=[florence2_tags_output1_1],
                )
                florence2_unload_btn1_1.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                florence2_send_to_txt2img1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("txt2img"))
                florence2_send_to_img2img1.click(fn=None, inputs=[], outputs=[], _js=get_send_js_code("img2img"))
            with gr.Tab(label=I18N["Image Output"][default_lang], elem_id="image_output"):
                with gr.Row():
                    # input
                    with gr.Column(variant="panel"):
                        input_image_florence2_1_2 = gr.Image(
                            label=I18N["Input Image"][default_lang],
                            sources=["upload", "webcam"],
                            type="pil",
                            height=520,
                            object_fit="contain",
                            elem_id="florence2_input_image1_2",
                        )
                        with gr.Row():
                            florence2_model_selector1_2 = gr.Dropdown(
                                label=I18N["Select Model"][default_lang],
                                choices=florence2_backend.model_list,
                                value=shared.opts.florence2_model,
                                elem_id="florence2_model_selector1_2",
                            )
                            florence2_lora_selector1_2 = gr.Dropdown(
                                label=I18N["Select Lora"][default_lang],
                                choices=florence2_backend.lora_list,
                                value=shared.opts.florence2_Lora,
                                elem_id="florence2_lora_selector1_2",
                            )
                        florence2_task1_2 = gr.Dropdown(
                            label=I18N["Select Task"][default_lang],
                            choices=["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"],
                            value=shared.opts.florence2_task,
                            elem_id="florence2_task_selector1_2",
                        )
                        florence2_text_input1_2 = gr.Textbox(
                            label=I18N["Text Input (for specific tasks)"][default_lang],
                            lines=2,
                            placeholder="Only for referring_expression_segmentation, caption_to_phrase_grounding, docvqa",
                            elem_id="florence2_text_input1_2",
                        )
                        with gr.Row():
                            florence2_num_beams1_2 = gr.Number(
                                label=I18N["Num Beams"][default_lang],
                                value=shared.opts.florence2_num_beams,
                                minimum=1,
                                maximum=10,
                                elem_id="florence2_num_beams_selector1_2",
                            )
                            florence2_max_token1_2 = gr.Number(
                                label=I18N["Max token"][default_lang],
                                value=shared.opts.florence2_max_token,
                                minimum=1,
                                maximum=4096,
                                elem_id="florence2_max_token_selector1_2",
                            )
                        with gr.Row():
                            florence2_dtype1_2 = gr.Dropdown(
                                label=I18N["Select Precision"][default_lang],
                                choices=florence2_backend.dtype,
                                value=shared.opts.florence2_dtype,
                                elem_id="florence2_dtype_selector1_2",
                            )
                            florence2_attention1_2 = gr.Dropdown(
                                label=I18N["Select Attention Implementation"][default_lang],
                                choices=florence2_backend.attention_list,
                                value=shared.opts.florence2_attention,
                                elem_id="florence2_attention_selector1_2",
                            )
                        with gr.Row():
                            florence2_fill_mask1 = gr.Checkbox(label=I18N["Fill Mask"][default_lang], value=shared.opts.florence2_fill_mask, elem_id="florence2_fill_mask_selector1_2")
                            florence2_keep_model_loaded1_2 = gr.Checkbox(label=I18N["Keep Model Loaded"][default_lang], value=shared.opts.florence2_keep_model_loaded, elem_id="florence2_keep_model_loaded_selector1_2")
                            florence2_show_json1_2 = gr.Checkbox(label=I18N["Show JSON"][default_lang], value=shared.opts.florence2_show_json_image, elem_id="florence2_is_show_json1_2")
                        with gr.Row():
                            florence2_interrogate_btn1_2 = gr.Button(value=I18N["Interrogate"][default_lang], variant="primary", elem_id="florence2_interrogate_btn1_2")
                            florence2_unload_btn1_2 = gr.Button(value=I18N["Unload Model"][default_lang], variant="secondary", elem_id="florence2_unload_btn1_2")
                    # output
                    with gr.Column(variant="panel"):
                        florence2_output_img1 = gr.Image(label=I18N["Output Image"][default_lang], type="pil", height=600, object_fit="contain", elem_id="florence2_output_image1", interactive=False)
                        florence2_output_mask_img1 = gr.Image(label=I18N["Output Mask Image"][default_lang], type="pil", height=600, object_fit="contain", elem_id="florence2_output_mask_image1", interactive=False)
                        with gr.Row():
                            florence2_save_output_img_btn = gr.Button(I18N["Save Output Image"][default_lang], variant="secondary", elem_id="florence2_save_output_img_btn1")
                            florence2_save_mask_img_btn = gr.Button(I18N["Save Mask Image"][default_lang], variant="secondary", elem_id="florence2_save_mask_img_btn1")
                        florence2_tags_output1_2 = gr.Textbox(label=I18N["Extracted Tags"][default_lang], lines=5, interactive=False, elem_id="florence2_extracted_tags1_2")
                    # 事件绑定
                    florence2_interrogate_btn1_2.click(
                        fn=partial(florence2_backend.predict, output_type="image"),
                        inputs=[input_image_florence2_1_2, florence2_model_selector1_2, florence2_lora_selector1_2, florence2_task1_2, florence2_text_input1_2, florence2_num_beams1_2, florence2_max_token1_2, florence2_dtype1_2, florence2_attention1_2, florence2_keep_model_loaded1_2, florence2_show_json1_2, florence2_fill_mask1],
                        outputs=[florence2_output_img1, florence2_output_mask_img1, florence2_tags_output1_2],
                    )
                    florence2_unload_btn1_2.click(fn=florence2_backend.unload_model, inputs=[], outputs=[])
                    florence2_save_output_img_btn.click(fn=partial(save_image, img_type="output"), inputs=[florence2_output_img1], outputs=[])
                    florence2_save_mask_img_btn.click(fn=partial(save_image, img_type="mask"), inputs=[florence2_output_mask_img1], outputs=[])

        # --- 参数更新 ---
        tagger_interface.load(
            fn=sync_opts_to_components,
            inputs=[],
            outputs=[
                # florence2
                florence2_model_selector1_1,
                florence2_model_selector1_2,
                florence2_lora_selector1_1,
                florence2_lora_selector1_2,
                florence2_task1_1,
                florence2_task1_2,
                florence2_num_beams1_1,
                florence2_num_beams1_2,
                florence2_max_token1_1,
                florence2_max_token1_2,
                florence2_dtype1_1,
                florence2_dtype1_2,
                florence2_attention1_1,
                florence2_attention1_2,
                florence2_keep_model_loaded1_1,
                florence2_keep_model_loaded1_2,
                florence2_show_json1_1,
                florence2_show_json1_2,
                florence2_fill_mask1,
            ],
            show_progress="hidden",
        )

    return [(tagger_interface, "Florence2", "Florence2_tab")]


def on_ui_settings():
    # 插件专属分类：(唯一标识, 显示名称)
    section = ("Florence2_setting", "Florence2")
    # 语言
    shared.opts.add_option("Florence2_language", shared.OptionInfo("English", I18N["Language"][default_lang], gr.Dropdown, {"choices": ["English", "简体中文", "繁體中文"]}, section=section))
    # florence2模型
    shared.opts.add_option("florence2_model_text", shared.OptionInfo("microsoft/Florence-2-large-ft", f"florence text {I18N["Select Model"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.model_list}, section=section))
    shared.opts.add_option("florence2_model_image", shared.OptionInfo("microsoft/Florence-2-large-ft", f"florence image {I18N["Select Model"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.model_list}, section=section))
    # florence2 Lora
    shared.opts.add_option("florence2_Lora_text", shared.OptionInfo(None, f"florence text {I18N["Select Lora"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.lora_list}, section=section))
    shared.opts.add_option("florence2_Lora_image", shared.OptionInfo(None, f"florence text {I18N["Select Lora"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.lora_list}, section=section))
    # florence2 task
    shared.opts.add_option("florence2_task_text", shared.OptionInfo("caption", f"florence text {I18N["Select Task"][default_lang]}", gr.Dropdown, {"choices": ["caption", "detailed_caption", "more_detailed_caption", "ocr", "docvqa(need text_input)", "prompt_gen_tags", "prompt_gen_mixed_caption", "prompt_gen_analyze", "prompt_gen_mixed_caption_plus"]}, section=section))
    shared.opts.add_option("florence2_task_image", shared.OptionInfo("region_caption", f"florence text {I18N["Select Task"][default_lang]}", gr.Dropdown, {"choices": ["region_caption", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding(need text_input)", "referring_expression_segmentation(need text_input)", "ocr_with_region"]}, section=section))
    # florence2 num_beams
    shared.opts.add_option("florence2_num_beams_text", shared.OptionInfo(3, f"florence text {I18N["Num Beams"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 10}, section=section))
    shared.opts.add_option("florence2_num_beams_image", shared.OptionInfo(3, f"florence text {I18N["Num Beams"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 10}, section=section))
    # florence2 max_token
    shared.opts.add_option("florence2_max_token_text", shared.OptionInfo(1024, f"florence text {I18N["Max token"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 4096}, section=section))
    shared.opts.add_option("florence2_max_token_image", shared.OptionInfo(1024, f"florence text {I18N["Max token"][default_lang]}", gr.Number, {"minimum": 1, "maximum": 4096}, section=section))
    # florence2 dtype
    shared.opts.add_option("florence2_dtype_text", shared.OptionInfo("fp16", f"florence text {I18N["Select Precision"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.dtype}, section=section))
    shared.opts.add_option("florence2_dtype_image", shared.OptionInfo("fp16", f"florence text {I18N["Select Precision"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.dtype}, section=section))
    # florence2 attention
    shared.opts.add_option("florence2_attention_text", shared.OptionInfo("sdpa", f"florence text {I18N["Select Attention Implementation"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.attention_list}, section=section))
    shared.opts.add_option("florence2_attention_image", shared.OptionInfo("sdpa", f"florence text {I18N["Select Attention Implementation"][default_lang]}", gr.Dropdown, {"choices": florence2_backend.attention_list}, section=section))
    # florence2 keep_model_loaded
    shared.opts.add_option("florence2_keep_model_loaded_text", shared.OptionInfo(False, f"florence text {I18N["Keep Model Loaded"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("florence2_keep_model_loaded_image", shared.OptionInfo(False, f"florence text {I18N["Keep Model Loaded"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    # florence2 show json
    shared.opts.add_option("florence2_show_json_text", shared.OptionInfo(False, f"florence text {I18N["Show JSON"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("florence2_show_json_image", shared.OptionInfo(False, f"florence text {I18N["Show JSON"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))
    # florence2 fill_mask
    shared.opts.add_option("florence2_fill_mask", shared.OptionInfo(True, f"florence image {I18N["Fill Mask"][default_lang]}", gr.Checkbox, {"interactive": True}, section=section))


# ----------------------------------------------------------------
# 路徑設定
current_file_path = Path(os.path.abspath(__file__))
EXTENSION_DIR = current_file_path.parent.parent
SYS_DIR = EXTENSION_DIR.parent.parent
OUTPUT_DIR = SYS_DIR / "output" / "tagger_output"
print(f"[Tagger-all] OUTPUT_DIR: {OUTPUT_DIR}")
MODELS_DIR = SYS_DIR / "models" / "tagger_models"
print(f"[Tagger-all] model_dirs: {MODELS_DIR}")
FLORENCE2_DIR = OUTPUT_DIR / "florence2_output"

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(FLORENCE2_DIR):
    os.mkdir(FLORENCE2_DIR)

with open(EXTENSION_DIR / "language.json", "r", encoding="utf-8") as f1:
    I18N = json.load(f1)  # 语言

try:
    default_lang = shared.opts.Florence2_language
except:
    print("[Tagger-all] defalut English of the UI.")
    default_lang = "English"

# --main--
florence2_backend = Florence2(MODELS_DIR)

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
