from __future__ import annotations

import asyncio
import difflib
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from nicegui import ui

from panza.entities.instruction import Instruction, SnippetInstruction
from panza.interface.snippet_utils import (
    apply_review_decision,
    apply_review_decision_bulk,
    build_txt_file_tree,
    export_kept_snippets,
    load_snippet_tool_state,
    persist_review_state,
    read_preview_file,
    review_counts_text,
    split_file_into_review_dataset,
)
from panza.writer import PanzaWriter


EVAL_PAGE_SIZE = 30
TRAINING_PAGE_SIZE = 20
SORT_ORIGINAL = "Original order"
SORT_FIELD_LENGTH = "Field length"
SORT_BLEU = "BLEU score"
SORT_CHOICES = [SORT_ORIGINAL, SORT_FIELD_LENGTH, SORT_BLEU]
DIRECTION_ASC = "Ascending"
DIRECTION_DESC = "Descending"
DIRECTION_CHOICES = [DIRECTION_DESC, DIRECTION_ASC]
EDITABLE_FIELD_KEY = "editable_text"
DEFAULT_EVALUATION_FIELDS = ["prompt", "panza_response"]


class PanzaNiceGUI:
    def __init__(
        self,
        writer: PanzaWriter,
        pre_personalization_writer: PanzaWriter | None,
        use_pre_personalization_model: bool = True,
        host: str = "localhost",
        port: int = 8080,
        username: str | None = None,
        training_data_path: str | None = None,
    ):
        self.writer = writer
        # allow an explicit pre-personalization writer; fall back to the main writer
        self.pre_personalization_writer = pre_personalization_writer or writer
        self.host = host
        self.port = port
        self.username = username
        self.training_data_path_override = training_data_path
        self.logo_path = self._find_logo()
        self.training_data_path = self._find_training_data_file()
        self.default_evaluation_path = self._find_default_evaluation_file()
        self.available_models = self._find_available_models()
        self.selected_model_path = self._find_current_model_dir() or (
            self.available_models[0] if self.available_models else None
        )
        if self.selected_model_path is not None:
            self.default_evaluation_path = (
                self._find_evaluation_file_for_model(self.selected_model_path)
                or self.default_evaluation_path
            )

        self.snippet_tool_root = self._repo_root() / "panza_snippet_tool"
        self.snippets_path = self.snippet_tool_root / "data" / "snippets.json"
        self.review_state_path = self.snippet_tool_root / "data" / "review_state.json"
        self.kept_jsonl_path = self.snippet_tool_root / "data" / "kept_snippets.jsonl"
        self.kept_json_path = self.snippet_tool_root / "data" / "kept_snippets.json"
        self.review_snippets: List[Dict[str, Any]] = []
        self.review_state: Dict[str, Any] = {}
        self.review_docs_by_source: Dict[str, List[Dict[str, Any]]] = {}
        self.review_document_paths: List[str] = []
        self.review_doc_index: int = 0
        self.review_current_source: Optional[str] = None
        self.review_cards_container = None
        self.review_status_label = None
        self.review_counts_label = None
        self.review_source_label = None

        self.file_selector_root = Path("/Users/jen/Downloads/")
        if not self.file_selector_root.exists():
            self.file_selector_root = self._repo_root()
        self.file_tree_container = None
        self.selected_file_path_label = None
        self.selected_file_viewer = None
        self.selected_file_split_button = None
        self.selected_file_split_status_label = None
        self.selected_file_path: Optional[Path] = None

        self.prompt_input = None
        self.output_area = None
        self.generic_output_area = None
        self.status_label = None
        self.model_selector = None
        self.model_name_label = None
        self.model_loading_status_label = None

        self.training_data_status_label = None
        self.training_data_page_status_label = None
        self.training_data_records_container = None
        self.training_data_records: List[Dict[str, Any]] = []
        self.training_data_page_index = 0

        self.eval_file_input = None
        self.eval_fields_input = None
        self.eval_sort_by_select = None
        self.eval_sort_field_select = None
        self.eval_sort_direction_select = None
        self.eval_show_diff_checkbox = None
        self.eval_file_status_label = None
        self.eval_page_status_label = None
        self.eval_records_container = None
        self.eval_records: List[Dict[str, Any]] = []
        self.eval_fields: List[str] = DEFAULT_EVALUATION_FIELDS.copy()
        self.eval_page_index = 0
        self.eval_file_path: Optional[Path] = None

        self._start_server()

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _find_logo(self) -> str:
        logo_path = self._repo_root() / "panza_logo.png"
        if not logo_path.exists():
            raise FileNotFoundError(
                f"Could not find panza_logo.png at expected location: {logo_path}"
            )
        return str(logo_path)

    def _resolve_path(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = (self._repo_root() / resolved).resolve()
        return resolved

    def _find_user_name(self) -> Optional[str]:
        if self.username:
            return self.username

        retriever = getattr(getattr(self.writer, "prompt_builder", None), "retriever", None)
        index_name = getattr(retriever, "index_name", None)
        if index_name:
            return str(index_name)

        model_dir = self._find_requested_model_dir()
        if model_dir is not None and model_dir.name.startswith("panza_"):
            return model_dir.name.removeprefix("panza_").split("-", 1)[0]
        return None

    def _find_training_data_file(self) -> Optional[Path]:
        if self.training_data_path_override:
            return self._resolve_path(self.training_data_path_override)

        retriever = getattr(getattr(self.writer, "prompt_builder", None), "retriever", None)
        db_path = getattr(retriever, "db_path", None)
        if db_path:
            data_dir = self._resolve_path(db_path)
            direct_train_file = data_dir / "train.jsonl"
            if direct_train_file.exists():
                return direct_train_file

            username = self._find_user_name()
            if username:
                nested_train_file = data_dir / username / "train.jsonl"
                if nested_train_file.exists():
                    return nested_train_file

        username = self._find_user_name()
        if username:
            return self._repo_root() / "data" / username / "train.jsonl"
        return None

    def _find_default_evaluation_file(self) -> Optional[Path]:
        model_dir = self._find_requested_model_dir()
        if model_dir is not None:
            eval_file = model_dir / "professor_prompts_filled_outputs.json"
            if eval_file.exists():
                return eval_file

        # Fallback to first available model evaluation file
        for model_path in self._find_available_models():
            fallback_eval = model_path / "professor_prompts_filled_outputs.json"
            if fallback_eval.exists():
                return fallback_eval

        return None

    def _find_requested_model_dir(self) -> Optional[Path]:
        llm = getattr(self.writer, "llm", None)
        for attr in ("checkpoint", "name", "gguf_file"):
            value = getattr(llm, attr, None)
            if not value:
                continue
            path = self._resolve_path(str(value))
            if path.exists():
                return path if path.is_dir() else path.parent
        return None

    def _find_available_models(self) -> List[Path]:
        model_root = self._repo_root() / "checkpoints" / "models"
        if not model_root.exists():
            return []
        username = self._find_user_name()
        models = []
        for path in sorted(model_root.iterdir()):
            if not path.is_dir():
                continue
            if username is None or username in path.name:
                models.append(path)
        return models

    def _find_current_model_dir(self) -> Optional[Path]:
        llm = getattr(self.writer, "llm", None)
        if llm is None:
            return None
        if hasattr(llm, "checkpoint"):
            path = Path(str(getattr(llm, "checkpoint"))).expanduser()
            if not path.is_absolute():
                path = self._repo_root() / path
            if path.exists():
                return path if path.is_dir() else path.parent
        if hasattr(llm, "gguf_file"):
            path = Path(str(getattr(llm, "gguf_file"))).expanduser()
            if not path.is_absolute():
                path = self._repo_root() / path
            if path.exists():
                return path.parent
        return None

    def _find_evaluation_file_for_model(self, model_dir: Optional[Path]) -> Optional[Path]:
        if model_dir is None:
            return None
        candidate = model_dir / "professor_prompts_filled_outputs.json"
        return candidate if candidate.exists() else None

    def _reload_model(self, model_path: Path) -> None:
        llm = getattr(self.writer, "llm", None)
        if llm is None:
            raise RuntimeError("Writer does not have an LLM to reload.")

        if hasattr(llm, "checkpoint") and hasattr(llm, "_load_model_and_tokenizer"):
            llm.checkpoint = str(model_path)
            llm._load_model_and_tokenizer()
        elif hasattr(llm, "gguf_file") and hasattr(llm, "_load_model"):
            llm.gguf_file = str(model_path)
            llm._load_model()
        else:
            raise RuntimeError(
                "Current LLM type does not support runtime model switching."
            )

        self.selected_model_path = model_path
        if self.model_name_label is not None:
            self.model_name_label.text = model_path.name
            self.model_name_label.update()
        if self.model_loading_status_label is not None:
            self.model_loading_status_label.text = f"Loaded model: {model_path.name}"
            self.model_loading_status_label.update()

        new_eval_file = self._find_evaluation_file_for_model(model_path)
        if new_eval_file is not None:
            self.default_evaluation_path = new_eval_file
            if self.eval_file_input is not None:
                self.eval_file_input.value = str(new_eval_file)
                self.eval_file_input.update()
                self._load_evaluation_file()

    def _on_model_selected(self) -> None:
        if self.model_selector is None or not self.model_selector.value:
            return
        selected_name = self.model_selector.value
        selected_path = next(
            (path for path in self.available_models if path.name == selected_name),
            None,
        )
        if selected_path is None:
            return
        try:
            self._reload_model(selected_path)
        except Exception as exc:
            if self.model_loading_status_label is not None:
                self.model_loading_status_label.text = f"Could not load model: {exc}"
                self.model_loading_status_label.update()

    def _build_interface(self) -> None:
        #ui.title("Panza")
        ui.markdown("# Panza")
        ui.image(self.logo_path).style(
            "max-width: 240px; margin: 0 auto 24px; display: block;"
        )

        with ui.tabs().classes("w-full") as tabs:
            self.tabs = tabs
            model_selector_tab = ui.tab("Model selector")
            inference_tab = ui.tab("Inference")
            training_data_tab = ui.tab("Training data")
            self.review_tab = ui.tab("Add training data")
            automated_evaluation_tab = ui.tab("Automated evaluation")

        with ui.tab_panels(tabs, value=model_selector_tab).classes("w-full"):
            with ui.tab_panel(model_selector_tab):
                self._build_model_selector_tab()
            with ui.tab_panel(inference_tab):
                self._build_inference_tab()
            with ui.tab_panel(training_data_tab):
                self._build_training_data_tab()
            with ui.tab_panel(self.review_tab):
                self._build_review_tab()
            with ui.tab_panel(automated_evaluation_tab):
                self._build_automated_evaluation_tab()

    def _build_model_selector_tab(self) -> None:
        selected_name = (
            self.selected_model_path.name if self.selected_model_path else None
        )
        available_names = [path.name for path in self.available_models]
        if selected_name and selected_name not in available_names:
            available_names.insert(0, selected_name)

        ui.label("Select which model should be used for inference and automated evaluation.").classes(
            "text-sm text-gray-600"
        )
        self.model_name_label = ui.label(
            f"Current model: {selected_name or 'None'}"
        )
        self.model_loading_status_label = ui.label(
            "" if self.selected_model_path else "No model selected."
        )
        self.model_selector = ui.select(
            available_names,
            label="Available models",
            value=selected_name or (available_names[0] if available_names else ""),
            on_change=self._on_model_selected,
        ).classes("w-full")

        if not available_names:
            ui.markdown(
                "**No models were found in `checkpoints/models`." \
                " Ensure your model directories are present and your username matches the model name.**"
            )

    def _build_inference_tab(self) -> None:
        self.prompt_input = ui.input(
            label="Prompt",
            placeholder="Enter a prompt here...",
            value="",
        ).style("width: 100%; margin-bottom: 16px;")

        self.generic_output_area = (
            ui.textarea(
                label="Generic Output",
                placeholder="Output from pre-personalization model will appear here.",
                value="",
            )
            .props("readonly")
            .style("width: 100%; min-height: 160px; margin-bottom: 12px;")
        )

        self.output_area = (
            ui.textarea(
                label="Panza Output",
                placeholder="The model output will appear here after execution.",
                value="",
            )
            .props("readonly")
            .style("width: 100%; min-height: 240px; margin-bottom: 16px;")
        )

        ui.button("Run prompt", on_click=self.execute_prompt).style(
            "margin-bottom: 16px;"
        )
        self.status_label = ui.label("")

    def _build_training_data_tab(self) -> None:
        display_path = str(self.training_data_path or "")
        ui.label(f"File: {display_path or 'No training data file resolved.'}").classes(
            "text-sm text-gray-600"
        )

        with ui.row().classes("w-full items-center"):
            ui.button("Previous", on_click=self._previous_training_data_page)
            self.training_data_page_status_label = ui.label("")
            ui.button("Next", on_click=self._next_training_data_page)

        self.training_data_status_label = ui.label("")
        ui.button("Add more training data", on_click=self._open_review_tab).style(
            "margin-top: 12px;"
        )
        self.training_data_records_container = ui.column().classes("w-full gap-4")
        self._load_training_data()

    def _open_review_tab(self) -> None:
        if self.tabs is None or self.review_tab is None:
            return
        try:
            self.tabs.value = self.review_tab
        except Exception:
            pass
        try:
            self.tabs.set_value(self.review_tab)
        except Exception:
            pass
        try:
            self.tabs.update()
        except Exception:
            pass

    def _load_snippet_tool_state(self) -> None:
        (
            self.review_snippets,
            self.review_state,
            self.review_docs_by_source,
            self.review_document_paths,
        ) = load_snippet_tool_state(self.snippets_path, self.review_state_path)
        self.review_doc_index = 0
        self.review_current_source = None

    def _select_file(self, path: Path) -> None:
        self.selected_file_path = path
        content = read_preview_file(path)
        if self.selected_file_path_label is not None:
            self.selected_file_path_label.text = str(path)
            self.selected_file_path_label.update()
        if self.selected_file_viewer is not None:
            self.selected_file_viewer.value = content
            self.selected_file_viewer.update()
        if self.selected_file_split_button is not None:
            self.selected_file_split_button.visible = True
            self.selected_file_split_button.update()
        if self.selected_file_split_status_label is not None:
            self.selected_file_split_status_label.text = ""
            self.selected_file_split_status_label.update()

    def _set_selected_file_split_status(self, status: str) -> None:
        if self.selected_file_split_status_label is not None:
            self.selected_file_split_status_label.text = status
            self.selected_file_split_status_label.update()

    async def _split_selected_file_into_snippets(self) -> None:
        if self.selected_file_path is None:
            self._set_selected_file_split_status("Select a .txt file first.")
            return

        split_button = self.selected_file_split_button
        if split_button is not None:
            split_button.enabled = False
            split_button.update()

        await asyncio.sleep(0)
        try:
            _, _, created_count, added_count, stats = split_file_into_review_dataset(
                self.selected_file_path,
                self.review_snippets,
                self.review_state,
                self.snippets_path,
                self.review_state_path,
                snippets_per_file=200,
            )
            if created_count == 0:
                reason = ", ".join(f"{key}: {value}" for key, value in sorted(stats.items()))
                self._set_selected_file_split_status(
                    f"No snippets were created from this file. {reason or 'No reason reported.'}"
                )
                return

            self._load_snippet_tool_state()
            self._load_review_snippet()
            self._set_selected_file_split_status(
                f"Created {created_count} snippet(s); added {added_count} new snippet(s) to review."
            )
        except Exception as exc:
            self._set_selected_file_split_status(f"Could not split file into snippets: {exc}")
        finally:
            if split_button is not None:
                split_button.enabled = True
                split_button.update()

    def _render_file_tree_node(self, node: Dict[str, Any], container: Any, depth: int = 0) -> None:
        path = node["path"]
        if node["type"] == "directory":
            if depth > 0:
                with container:
                    ui.label(path.name).classes("font-semibold").style(f"margin-left: {depth * 16}px;")
            for child in node["children"]:
                self._render_file_tree_node(child, container, depth + 1)
        else:
            with container:
                ui.button(
                    path.name,
                    on_click=lambda selected_path=path: self._select_file(selected_path),
                ).props("flat").classes("w-full text-left").style(
                    f"margin-left: {depth * 16}px;"
                )

    def _load_file_tree(self) -> None:
        if self.file_tree_container is None:
            return
        self.file_tree_container.clear()
        if not self.file_selector_root.exists():
            ui.label(f"Directory not found: {self.file_selector_root}").classes("text-sm text-red-600")
            return
        file_tree = build_txt_file_tree(self.file_selector_root)
        if file_tree is None:
            ui.label("No .txt files found.").classes("text-sm text-gray-600")
            return
        self._render_file_tree_node(file_tree, self.file_tree_container)

    def _export_kept_snippets(self) -> None:
        export_kept_snippets(
            self.review_snippets,
            self.review_state,
            self.kept_jsonl_path,
            self.kept_json_path,
        )

    def _load_review_snippet(self) -> None:
        if self.review_cards_container is None:
            return

        self.review_cards_container.clear()
        total_docs = len(self.review_document_paths)
        total_snippets = len(self.review_snippets)

        if total_snippets == 0:
            with self.review_cards_container:
                ui.label("No candidate snippets to review.").classes("text-sm text-gray-600")
            if self.review_status_label is not None:
                self.review_status_label.text = "No snippets available."
                self.review_status_label.update()
            if self.review_counts_label is not None:
                self.review_counts_label.text = review_counts_text(self.review_snippets, self.review_state)
                self.review_counts_label.update()
            if self.review_source_label is not None:
                self.review_source_label.text = "0"
                self.review_source_label.update()
            return

        if self.review_source_label is not None:
            self.review_source_label.text = str(total_docs)
            self.review_source_label.update()
        if self.review_status_label is not None:
            self.review_status_label.text = f"Total snippets: {total_snippets}"
            self.review_status_label.update()
        if self.review_counts_label is not None:
            self.review_counts_label.text = review_counts_text(self.review_snippets, self.review_state)
            self.review_counts_label.update()

        with self.review_cards_container:
            for source_path in self.review_document_paths:
                snippets = self.review_docs_by_source.get(source_path, [])
                with ui.card().classes("w-full p-4 mb-4"):
                    ui.markdown(f"### {source_path}")
                    ui.label(f"{len(snippets)} candidate snippet(s)").classes("text-sm text-gray-600")
                    with ui.row().classes("w-full gap-4 items-center").style("margin-top: 12px;"):
                        async def keep_all_click(current_source=source_path):
                            keep_all_button.enabled = False
                            keep_all_button.update()
                            await asyncio.sleep(0)
                            await self._review_action_bulk("keep", current_source)

                        async def delete_all_click(current_source=source_path):
                            delete_all_button.enabled = False
                            delete_all_button.update()
                            await asyncio.sleep(0)
                            await self._review_action_bulk("delete", current_source)

                        keep_all_button = ui.button(
                            "Keep all",
                            on_click=keep_all_click,
                        ).props("color=positive")
                        delete_all_button = ui.button(
                            "Delete all",
                            on_click=delete_all_click,
                        ).props("color=negative")
                        ui.label("").style("flex:1")
                    with ui.column().classes("w-full gap-4 mt-4"):
                        for snippet in snippets:
                            snippet_id = str(snippet.get("id", ""))
                            decision = self.review_state.get("decisions", {}).get(snippet_id, "pending")
                            card_style = ""
                            if decision == "keep":
                                card_style = "background-color: #e6ffed;"
                            elif decision == "delete":
                                card_style = "background-color: #ffe4ec;"
                            with ui.card().classes("w-full p-4").style(card_style) as card:
                                ui.markdown(f"**Snippet ID:** {snippet_id or 'unknown'}")
                                ui.markdown(f"**Decision:** {decision}")
                                ui.textarea(
                                    value=str(snippet.get("snippet_text", "")),
                                ).props("readonly").classes("w-full").style(
                                    "min-height: 120px; white-space: pre-wrap; margin-bottom: 8px;"
                                )
                                with ui.row().classes("w-full gap-4 items-center"):
                                    async def keep_click(current_id=snippet_id, snippet_card=card):
                                        snippet_card.style("background-color: #d3d3d3;")
                                        snippet_card.update()
                                        keep_button.enabled = False
                                        keep_button.update()
                                        await asyncio.sleep(0)
                                        await self._review_action("keep", current_id)

                                    async def delete_click(current_id=snippet_id, snippet_card=card):
                                        snippet_card.style("background-color: #d3d3d3;")
                                        snippet_card.update()
                                        delete_button.enabled = False
                                        delete_button.update()
                                        await asyncio.sleep(0)
                                        await self._review_action("delete", current_id)

                                    keep_button = ui.button(
                                        "Keep",
                                        on_click=keep_click,
                                    ).props("color=positive")
                                    delete_button = ui.button(
                                        "Delete",
                                        on_click=delete_click,
                                    ).props("color=negative")

    async def _review_action(self, decision: str, snippet_id: str) -> None:
        if not snippet_id:
            return
        apply_review_decision(self.review_state, snippet_id, decision)
        persist_review_state(self.review_state_path, self.review_state)
        self._export_kept_snippets()
        self._load_review_snippet()

    async def _review_action_bulk(self, decision: str, source_path: str) -> None:
        if not source_path:
            return
        snippets = self.review_docs_by_source.get(source_path, [])
        apply_review_decision_bulk(self.review_state, snippets, decision)
        persist_review_state(self.review_state_path, self.review_state)
        self._export_kept_snippets()
        self._load_review_snippet()

    def _build_review_tab(self) -> None:
        self._load_snippet_tool_state()
        ui.label("File selector").classes("text-sm text-gray-600")
        with ui.row().classes("w-full gap-4").style("margin-bottom: 16px; align-items: flex-start;"):
            with ui.column().classes("w-1/3").style("max-height: 420px; overflow:auto; border:1px solid #ddd; padding: 12px; background:#fafafa;"):
                ui.label(f"Base directory: {self.file_selector_root}").classes("text-sm text-gray-600")
                self.file_tree_container = ui.column().classes("w-full gap-2")
                self._load_file_tree()
            with ui.column().classes("w-2/3").style("min-height: 420px;"):
                ui.label("Selected file preview").classes("text-sm text-gray-600")
                self.selected_file_path_label = ui.label("No file selected").classes("text-sm text-gray-600")
                self.selected_file_viewer = ui.textarea(
                    value="",
                ).props("readonly").style("width: 100%; min-height: 360px;")
                self.selected_file_split_button = ui.button(
                    "Split into snippets",
                    on_click=self._split_selected_file_into_snippets,
                ).props("color=primary")
                self.selected_file_split_button.visible = False
                self.selected_file_split_status_label = ui.label("").classes("text-sm text-gray-600")

        ui.label("Review candidate snippets and keep the ones you want to add to training data. All candidates are shown below grouped by document path.").classes(
            "text-sm text-gray-600"
        )
        self.review_status_label = ui.label("")
        self.review_counts_label = ui.label("")
        self.review_source_label = ui.label("")
        with ui.row().classes("w-full gap-4 items-center").style("margin-bottom: 12px;"):
            ui.button("Export kept snippets", on_click=self._export_kept_snippets).props("color=secondary")
            ui.label("").style("flex:1")
            self.review_counts_label = ui.label("")
        with ui.row().classes("w-full gap-4 items-center").style("margin-bottom: 12px;"):
            ui.label("Documents:").classes("font-semibold")
            self.review_source_label = ui.label("")
            ui.label("Total snippets:").classes("font-semibold")
            self.review_status_label = ui.label("")
        self.review_cards_container = ui.column().classes("w-full gap-4")
        self._load_review_snippet()
        with ui.row().classes("w-full gap-4 items-center").style("margin-top: 12px;"):
            ui.button("Export kept snippets", on_click=self._export_kept_snippets).props("color=secondary")

    def _build_automated_evaluation_tab(self) -> None:
        default_path = str(self.default_evaluation_path or "")
        self.eval_file_input = ui.input(
            label="Evaluation file",
            value=default_path,
        ).classes("w-full")
        self.eval_fields_input = ui.input(
            label="Fields",
            value=", ".join(DEFAULT_EVALUATION_FIELDS),
        ).classes("w-full")

        with ui.row().classes("w-full items-end"):
            self.eval_sort_by_select = ui.select(
                SORT_CHOICES,
                label="Sort by",
                value=SORT_ORIGINAL,
                on_change=self._render_evaluation_records,
            )
            self.eval_sort_field_select = ui.select(
                DEFAULT_EVALUATION_FIELDS,
                label="Length field",
                value=DEFAULT_EVALUATION_FIELDS[0],
                on_change=self._render_evaluation_records,
            )
            self.eval_sort_direction_select = ui.select(
                DIRECTION_CHOICES,
                label="Direction",
                value=DIRECTION_DESC,
                on_change=self._render_evaluation_records,
            )
            self.eval_show_diff_checkbox = ui.checkbox(
                "Show diffs",
                value=True,
                on_change=self._render_evaluation_records,
            )
            ui.button("Load", on_click=self._load_evaluation_file)

        with ui.row().classes("w-full items-center"):
            ui.button("Previous", on_click=self._previous_evaluation_page)
            self.eval_page_status_label = ui.label("")
            ui.button("Next", on_click=self._next_evaluation_page)

        self.eval_file_status_label = ui.label("")
        self.eval_records_container = ui.column().classes("w-full gap-4")
        self._load_evaluation_file()

    def _predict(self, input: str) -> Generator:
        instruction: Instruction = SnippetInstruction(input, context="")
        stream: Generator = self.writer.run(instruction, stream=True)
        return stream

    def _predict_with_writer(self, writer: PanzaWriter, input: str) -> Generator:
        instruction: Instruction = SnippetInstruction(input, context="")
        stream: Generator = writer.run(instruction, stream=True)
        return stream

    def _streamer(self, stream):
        for chunk in stream:
            yield chunk

    async def execute_prompt(self) -> None:
        prompt = (self.prompt_input.value or "").strip()
        if not prompt:
            self.status_label.text = "Please enter a prompt before running."
            self.status_label.update()
            return

        self.status_label.text = "Running prompt..."
        # clear both output areas
        if self.generic_output_area is not None:
            self.generic_output_area.value = ""
            self.generic_output_area.update()
        self.output_area.value = ""
        self.status_label.update()
        self.output_area.update()
        await asyncio.sleep(0)

        # Stage 1: run pre-personalization writer to get generic output
        generic_output = ""
        pre_writer = self.pre_personalization_writer
        stream1 = self._predict_with_writer(pre_writer, prompt)
        for chunk in self._streamer(stream1):
            generic_output += chunk
            if self.generic_output_area is not None:
                self.generic_output_area.value = generic_output
                self.generic_output_area.update()
            await asyncio.sleep(0)

        # Stage 2: run main writer on the generic output and show Panza Output
        panza_output = ""
        stream2 = self._predict_with_writer(self.writer, generic_output or prompt)
        for chunk in self._streamer(stream2):
            panza_output += chunk
            self.output_area.value = panza_output
            self.output_area.update()
            await asyncio.sleep(0)

        self.status_label.text = "Prompt complete."
        self.status_label.update()

    def _read_jsonl_records(self, file_path: Path) -> List[Dict[str, Any]]:
        records = []
        with file_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_line_number"] = line_number
                record["_record_number"] = len(records) + 1
                records.append(record)
        return records

    def _load_training_data(self) -> None:
        self.training_data_records = []
        self.training_data_page_index = 0

        if self.training_data_path is None:
            self._set_training_data_status("No training data file could be resolved.")
            self._render_training_data_records()
            return

        if not self.training_data_path.exists():
            self._set_training_data_status(
                f"Training data file not found: {self.training_data_path}"
            )
            self._render_training_data_records()
            return

        try:
            self.training_data_records = self._read_jsonl_records(self.training_data_path)
        except Exception as exc:
            self._set_training_data_status(f"Could not load training data: {exc}")
            self._render_training_data_records()
            return

        self._set_training_data_status(
            f"Loaded {len(self.training_data_records)} records from {self.training_data_path}"
        )
        self._render_training_data_records()

    def _set_training_data_status(self, status: str) -> None:
        if self.training_data_status_label is not None:
            self.training_data_status_label.text = status
            self.training_data_status_label.update()

    def _get_training_data_page_records(self) -> Tuple[List[Dict[str, Any]], str, int]:
        if not self.training_data_records:
            return [], "No training data loaded.", 0

        page_count = max(
            1,
            (len(self.training_data_records) + TRAINING_PAGE_SIZE - 1)
            // TRAINING_PAGE_SIZE,
        )
        self.training_data_page_index = max(
            0,
            min(self.training_data_page_index, page_count - 1),
        )
        start_index = self.training_data_page_index * TRAINING_PAGE_SIZE
        end_index = min(start_index + TRAINING_PAGE_SIZE, len(self.training_data_records))
        status = (
            f"Page {self.training_data_page_index + 1} of {page_count} | "
            f"Showing {start_index + 1}-{end_index} of {len(self.training_data_records)}"
        )
        return self.training_data_records[start_index:end_index], status, start_index

    def _training_record_title(self, record: Dict[str, Any], index: int) -> str:
        record_number = record.get("_record_number", index + 1)
        title = f"### Record {record_number}"
        record_id = record.get("id")
        if record_id:
            title += f" | {record_id}"
        source_path = record.get("source_path")
        if source_path:
            title += f" | {Path(str(source_path)).name}"
        return title

    def _render_training_data_field(self, field: str, record: Dict[str, Any]) -> None:
        if field not in record:
            return
        ui.textarea(
            label=field,
            value=self._display_value(record.get(field)),
        ).props("readonly").classes("w-full").style("min-height: 96px;")

    def _render_training_data_records(self) -> None:
        if self.training_data_records_container is None:
            return

        page_records, status, page_start = self._get_training_data_page_records()
        if self.training_data_page_status_label is not None:
            self.training_data_page_status_label.text = status
            self.training_data_page_status_label.update()

        self.training_data_records_container.clear()
        with self.training_data_records_container:
            if not page_records:
                ui.label("No training records to display.")
                return

            preferred_fields = [
                "summary",
                "snippet_text",
                "original_text",
                "source_path",
                "snippet_word_count",
                "paragraph_count",
            ]
            for slot, record in enumerate(page_records):
                with ui.column().classes("w-full gap-2").style(
                    "border-bottom: 1px solid #e5e7eb; padding: 12px 0;"
                ):
                    ui.markdown(self._training_record_title(record, page_start + slot))
                    rendered_fields = [
                        field for field in preferred_fields if field in record
                    ]
                    if rendered_fields:
                        for field in rendered_fields:
                            self._render_training_data_field(field, record)
                    else:
                        ui.code(
                            json.dumps(record, ensure_ascii=False, indent=2),
                            language="json",
                        ).classes("w-full")

    def _previous_training_data_page(self) -> None:
        self.training_data_page_index -= 1
        self._render_training_data_records()

    def _next_training_data_page(self) -> None:
        self.training_data_page_index += 1
        self._render_training_data_records()

    def _parse_evaluation_fields(self) -> List[str]:
        raw_fields = self.eval_fields_input.value if self.eval_fields_input else ""
        fields = [field.strip() for field in raw_fields.split(",") if field.strip()]
        return fields or DEFAULT_EVALUATION_FIELDS.copy()

    def _read_evaluation_records(self, file_path: Path) -> List[Dict[str, Any]]:
        if file_path.suffix.lower() == ".jsonl":
            records = []
            with file_path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["_line_number"] = line_number
                    record["_record_number"] = len(records) + 1
                    records.append(record)
            return records

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        records = data["responses"] if isinstance(data, dict) and "responses" in data else data
        if not isinstance(records, list):
            raise ValueError("Expected a JSON list or an object with a 'responses' list.")

        for index, record in enumerate(records):
            record["_line_number"] = index
            record["_record_number"] = index + 1
            if "panza_responses" in record and "panza_response" not in record:
                record["panza_response"] = record["panza_responses"][0]
        return records

    def _write_evaluation_records(self) -> None:
        if self.eval_file_path is None:
            return

        clean_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in self.eval_records
        ]
        if self.eval_file_path.suffix.lower() == ".jsonl":
            with self.eval_file_path.open("w", encoding="utf-8") as file:
                for record in clean_records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            return

        with self.eval_file_path.open("r", encoding="utf-8") as file:
            original_data = json.load(file)
        if isinstance(original_data, dict) and "responses" in original_data:
            original_data["responses"] = clean_records
            output_data = original_data
        else:
            output_data = clean_records
        with self.eval_file_path.open("w", encoding="utf-8") as file:
            json.dump(output_data, file, ensure_ascii=False, indent=2)

    def _load_evaluation_file(self) -> None:
        if self.eval_file_input is None:
            return

        file_path = Path(self.eval_file_input.value or "").expanduser()
        if not file_path.is_absolute():
            file_path = (self._repo_root() / file_path).resolve()

        self.eval_records = []
        self.eval_file_path = file_path
        self.eval_page_index = 0

        if not file_path.exists():
            self._set_evaluation_status(f"Evaluation file not found: {file_path}")
            self._render_evaluation_records()
            return

        try:
            self.eval_records = self._read_evaluation_records(file_path)
            self.eval_fields = self._parse_evaluation_fields()
        except Exception as exc:
            self._set_evaluation_status(f"Could not load evaluation file: {exc}")
            self._render_evaluation_records()
            return

        available_fields = set()
        for record in self.eval_records:
            available_fields.update(record.keys())
        missing_fields = [field for field in self.eval_fields if field not in available_fields]
        if self.eval_sort_field_select is not None:
            self.eval_sort_field_select.set_options(self.eval_fields, value=self.eval_fields[0])
            self.eval_sort_field_select.update()

        status = f"Loaded {len(self.eval_records)} records from {file_path}"
        if missing_fields:
            status += f" | Missing fields: {', '.join(missing_fields)}"
        self._set_evaluation_status(status)
        self._render_evaluation_records()

    def _set_evaluation_status(self, status: str) -> None:
        if self.eval_file_status_label is not None:
            self.eval_file_status_label.text = status
            self.eval_file_status_label.update()

    def _display_value(self, value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value)

    def _get_bleu_score(self, record: Dict[str, Any]) -> Optional[str]:
        scores = record.get("scores")
        if not isinstance(scores, dict) or "BLEU" not in scores:
            return None
        bleu = scores["BLEU"]
        if isinstance(bleu, list):
            return ", ".join(str(value) for value in bleu) if bleu else None
        return str(bleu)

    def _get_bleu_value(self, record: Dict[str, Any]) -> Optional[float]:
        scores = record.get("scores")
        if not isinstance(scores, dict) or "BLEU" not in scores:
            return None
        bleu = scores["BLEU"]
        if isinstance(bleu, list):
            bleu = bleu[0] if bleu else None
        try:
            return float(bleu)
        except (TypeError, ValueError):
            return None

    def _get_field_length(self, record: Dict[str, Any], field: str) -> int:
        return len(self._display_value(record.get(field, "")))

    def _sorted_evaluation_records(self) -> List[Dict[str, Any]]:
        sort_by = self.eval_sort_by_select.value if self.eval_sort_by_select else SORT_ORIGINAL
        sort_field = (
            self.eval_sort_field_select.value
            if self.eval_sort_field_select and self.eval_sort_field_select.value in self.eval_fields
            else self.eval_fields[0]
        )
        sort_direction = (
            self.eval_sort_direction_select.value
            if self.eval_sort_direction_select
            else DIRECTION_DESC
        )
        reverse = sort_direction == DIRECTION_DESC

        if sort_by == SORT_FIELD_LENGTH:
            return sorted(
                self.eval_records,
                key=lambda record: self._get_field_length(record, sort_field),
                reverse=reverse,
            )

        if sort_by == SORT_BLEU:
            records_with_bleu = [
                record for record in self.eval_records if self._get_bleu_value(record) is not None
            ]
            records_without_bleu = [
                record for record in self.eval_records if self._get_bleu_value(record) is None
            ]
            return sorted(
                records_with_bleu,
                key=lambda record: self._get_bleu_value(record),
                reverse=reverse,
            ) + records_without_bleu

        return self.eval_records

    def _get_evaluation_page_records(self) -> Tuple[List[Dict[str, Any]], str, int]:
        sorted_records = self._sorted_evaluation_records()
        if not sorted_records:
            return [], "No records loaded.", 1

        page_count = max(1, (len(sorted_records) + EVAL_PAGE_SIZE - 1) // EVAL_PAGE_SIZE)
        self.eval_page_index = max(0, min(self.eval_page_index, page_count - 1))
        start_index = self.eval_page_index * EVAL_PAGE_SIZE
        end_index = min(start_index + EVAL_PAGE_SIZE, len(sorted_records))
        status = (
            f"Page {self.eval_page_index + 1} of {page_count} | "
            f"Showing {start_index + 1}-{end_index} of {len(sorted_records)}"
        )
        return sorted_records[start_index:end_index], status, start_index

    def _highlight_diff_values(self, value_a: str, value_b: str) -> Tuple[str, str]:
        tokens_a = re.findall(r"\s+|\S+", value_a)
        tokens_b = re.findall(r"\s+|\S+", value_b)
        matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)
        highlighted_a = []
        highlighted_b = []

        for tag, start_a, end_a, start_b, end_b in matcher.get_opcodes():
            chunk_a = html.escape("".join(tokens_a[start_a:end_a]))
            chunk_b = html.escape("".join(tokens_b[start_b:end_b]))
            if tag == "equal":
                highlighted_a.append(chunk_a)
                highlighted_b.append(chunk_b)
            elif tag == "delete":
                highlighted_a.append(
                    f'<strong style="background:#ffe3e3;color:#9f1239;">{chunk_a}</strong>'
                )
            elif tag == "insert":
                highlighted_b.append(
                    f'<strong style="background:#dcfce7;color:#166534;">{chunk_b}</strong>'
                )
            else:
                highlighted_a.append(
                    f'<strong style="background:#ffe3e3;color:#9f1239;">{chunk_a}</strong>'
                )
                highlighted_b.append(
                    f'<strong style="background:#dcfce7;color:#166534;">{chunk_b}</strong>'
                )

        return "".join(highlighted_a), "".join(highlighted_b)

    def _render_diff_field(self, field: str, value_html: str) -> None:
        ui.html(
            '<div style="border:1px solid #d1d5db;border-radius:6px;padding:8px;'
            'background:#fff;min-height:96px;">'
            '<div style="font-size:12px;font-weight:600;color:#374151;'
            f'margin-bottom:6px;">{html.escape(field)}</div>'
            '<div style="white-space:pre-wrap;font-family:ui-monospace,'
            'SFMono-Regular,Menlo,Consolas,monospace;'
            f'font-size:13px;line-height:1.35;color:#111827;">{value_html}</div>'
            "</div>",
            sanitize=False,
        ).classes("w-full")

    def _render_readonly_field(self, field: str, value: Any) -> None:
        ui.textarea(
            label=field,
            value=self._display_value(value),
        ).props("readonly").classes("w-full").style("min-height: 96px;")

    def _render_evaluation_records(self) -> None:
        if self.eval_records_container is None:
            return

        page_records, status, page_start = self._get_evaluation_page_records()
        if self.eval_page_status_label is not None:
            self.eval_page_status_label.text = status
            self.eval_page_status_label.update()

        self.eval_records_container.clear()
        with self.eval_records_container:
            if not page_records:
                ui.label("No evaluation records to display.")
                return

            show_diff = (
                bool(self.eval_show_diff_checkbox.value)
                if self.eval_show_diff_checkbox is not None
                else True
            )
            diff_field_a = self.eval_fields[0] if self.eval_fields else None
            diff_field_b = self.eval_fields[1] if len(self.eval_fields) > 1 else None

            for slot, record in enumerate(page_records):
                sorted_index = page_start + slot
                record_number = record.get("_record_number", sorted_index + 1)
                with ui.column().classes("w-full gap-2").style(
                    "border-bottom: 1px solid #e5e7eb; padding: 12px 0;"
                ):
                    ui.markdown(f"### Record {record_number} | Result {sorted_index + 1}")
                    bleu_score = self._get_bleu_score(record)
                    if bleu_score is not None:
                        ui.markdown(f"**BLEU:** {bleu_score}")

                    value_a = self._display_value(record.get("prompt", "N/A"))
                    value_b = self._display_value(record.get("panza_response", "N/A"))
                    with ui.row().classes("w-full items-stretch gap-4"):
                        with ui.column().classes("w-full"):
                            if show_diff:
                                highlighted_a, highlighted_b = self._highlight_diff_values(value_a, value_b)
                                self._render_diff_field("prompt", highlighted_a)
                            else:
                                self._render_readonly_field("prompt", record.get("prompt", "N/A"))
                        with ui.column().classes("w-full"):
                            if show_diff:
                                self._render_diff_field("panza_response", self._highlight_diff_values(value_a, value_b)[1])
                            else:
                                self._render_readonly_field("panza_response", record.get("panza_response", "N/A"))

                    for field in self.eval_fields:
                        if field in {"prompt", "panza_response"}:
                            continue
                        self._render_readonly_field(field, record.get(field, "N/A"))

                    note_area = ui.textarea(
                        label="Notes",
                        value=self._display_value(record.get(EDITABLE_FIELD_KEY, "")),
                    ).classes("w-full").style("min-height: 72px;")
                    ui.button(
                        "Save note",
                        on_click=lambda current_record=record, textarea=note_area: (
                            self._save_evaluation_note(current_record, textarea.value)
                        ),
                    )

    def _save_evaluation_note(self, record: Dict[str, Any], value: str) -> None:
        record[EDITABLE_FIELD_KEY] = value
        self._write_evaluation_records()
        self._set_evaluation_status(f"Saved note to {self.eval_file_path}")

    def _previous_evaluation_page(self) -> None:
        self.eval_page_index -= 1
        self._render_evaluation_records()

    def _next_evaluation_page(self) -> None:
        self.eval_page_index += 1
        self._render_evaluation_records()

    def _start_server(self) -> None:
        ui.run(root=self._build_interface, host=self.host, port=self.port, reload=False)
