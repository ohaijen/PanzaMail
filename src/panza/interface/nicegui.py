from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Generator

from nicegui import ui
# try:
#     from src.panza.interface.nicegui import ui
# except ImportError as exc:
#     raise ImportError(
#         "NiceGUI is required for this interface. Install with `pip install nicegui`."
#     ) from exc

from panza.entities.instruction import Instruction, SnippetInstruction
from panza.writer import PanzaWriter


class PanzaNiceGUI:
    def __init__(self, writer: PanzaWriter, host: str = "localhost", port: int = 8080):
        self.writer = writer
        self.host = host
        self.port = port
        self.logo_path = self._find_logo()
        self.prompt_input = None
        self.output_area = None
        self.status_label = None
        self._start_server()

    def _find_logo(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        logo_path = repo_root / "../panza_logo.png"
        if not logo_path.exists():
            raise FileNotFoundError(
                f"Could not find panza_logo.png at expected location: {logo_path}"
            )
        return str(logo_path)

    def _build_interface(self) -> None:
        ui.markdown("# Panza")
        ui.image(self.logo_path).style(
            "max-width: 240px; margin: 0 auto 24px; display: block;"
        )

        self.prompt_input = ui.input(
            label="Prompt",
            placeholder="Enter a prompt here...",
            value="",
        ).style("width: 100%; margin-bottom: 16px;")

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

    def _predict(self, input: str) -> Generator:
        instruction: Instruction = SnippetInstruction(input, context="")
        stream: Generator = self.writer.run(instruction, stream=True)
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
        self.output_area.value = ""
        self.status_label.update()
        self.output_area.update()
        await asyncio.sleep(0)

        output = ""
        stream = self._predict(prompt)
        for chunk in self._streamer(stream):
            output += chunk
            self.output_area.value = output
            self.output_area.update()
            await asyncio.sleep(0)

        self.status_label.text = "Prompt complete."
        self.status_label.update()

    def _start_server(self) -> None:
        ui.run(root=self._build_interface, host=self.host, port=self.port, reload=False)
