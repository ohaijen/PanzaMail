#!/usr/bin/env python3
"""Create a Gradio interface for viewing JSONL/JSON records."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import gradio as gr
except ImportError:
    print("Gradio is required. Install with: pip install gradio")
    exit(1)


PAGE_SIZE = 25
SORT_ORIGINAL = "Original order"
SORT_FIELD_LENGTH = "Field length"
SORT_BLEU = "BLEU score"
SORT_CHOICES = [SORT_ORIGINAL, SORT_FIELD_LENGTH, SORT_BLEU]
DIRECTION_ASC = "Ascending"
DIRECTION_DESC = "Descending"
DIRECTION_CHOICES = [DIRECTION_DESC, DIRECTION_ASC]


def read_json_records(file_path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSON file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        all_records = json.load(f)
    assert "responses" in all_records
    records = all_records["responses"]
    for line_num, record in enumerate(records):
        record['_line_number'] = line_num
        record['_record_number'] = line_num + 1
        if 'panza_responses' in record:
            record['panza_response'] = record['panza_responses'][0]
    return records



def read_jsonl_records(file_path: str) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Add line number for tracking
                record['_line_number'] = line_num
                record['_record_number'] = len(records) + 1
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
    return records


# def write_jsonl_records(file_path: str, records: List[Dict[str, Any]]) -> None:
#     """Write records back to JSONL file, preserving order."""
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for record in records:
#             # Remove our internal tracking field
#             clean_record = {k: v for k, v in record.items() if k != '_line_number'}
#             f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')


def is_json_file(file_path: str) -> bool:
    """Return whether the input path points to a JSON file."""
    return Path(file_path).suffix.lower() == ".json"


def is_jsonl_file(file_path: str) -> bool:
    """Return whether the input path points to a JSONL file."""
    return Path(file_path).suffix.lower() == ".jsonl"


def get_bleu_score(record: Dict[str, Any]) -> Optional[str]:
    """Return a displayable BLEU score if the record has one."""
    scores = record.get("scores")
    if not isinstance(scores, dict) or "BLEU" not in scores:
        return None

    bleu = scores["BLEU"]
    if isinstance(bleu, list):
        if not bleu:
            return None
        return ", ".join(str(value) for value in bleu)
    return str(bleu)


def get_bleu_value(record: Dict[str, Any]) -> Optional[float]:
    """Return a numeric BLEU score for sorting."""
    scores = record.get("scores")
    if not isinstance(scores, dict) or "BLEU" not in scores:
        return None

    bleu = scores["BLEU"]
    if isinstance(bleu, list):
        if not bleu:
            return None
        bleu = bleu[0]

    try:
        return float(bleu)
    except (TypeError, ValueError):
        return None


def display_value(value: Any) -> str:
    """Return a stable display string for field values."""
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def get_field_length(record: Dict[str, Any], field: str) -> int:
    """Return the display-string length for a selected field."""
    return len(display_value(record.get(field, "")))


def sort_records(
    records: List[Dict[str, Any]],
    sort_by: str,
    sort_field: str,
    sort_direction: str,
) -> List[Dict[str, Any]]:
    """Sort records according to the selected controls."""
    reverse = sort_direction == DIRECTION_DESC

    if sort_by == SORT_FIELD_LENGTH:
        return sorted(
            records,
            key=lambda record: get_field_length(record, sort_field),
            reverse=reverse,
        )

    if sort_by == SORT_BLEU:
        records_with_bleu = [
            record for record in records if get_bleu_value(record) is not None
        ]
        records_without_bleu = [
            record for record in records if get_bleu_value(record) is None
        ]
        return sorted(
            records_with_bleu,
            key=lambda record: get_bleu_value(record),
            reverse=reverse,
        ) + records_without_bleu

    return records


def format_record_header(
    record: Dict[str, Any],
    sorted_index: int,
    sort_by: str,
    sort_field: str,
) -> str:
    """Return the row heading for a record."""
    record_number = record.get("_record_number", sorted_index + 1)
    header = f"### Record {record_number} | Result {sorted_index + 1}"

    if sort_by == SORT_FIELD_LENGTH:
        header += f" | {sort_field} length: {get_field_length(record, sort_field)}"

    return header


def create_main_interface(jsonl_file: str, fields: List[str]):
    """Create the main Gradio interface."""
    if is_jsonl_file(jsonl_file):
        records = read_jsonl_records(jsonl_file)
    else:
        records = read_json_records(jsonl_file)

    if not records:
        raise ValueError(f"No valid records found in '{jsonl_file}'.")

    show_bleu = is_json_file(jsonl_file)
    default_sort_field = fields[0]

    def get_page_count(sorted_records: List[Dict[str, Any]]) -> int:
        return max(1, (len(sorted_records) + PAGE_SIZE - 1) // PAGE_SIZE)

    def get_page_records(
        page_index: int,
        sort_by: str,
        sort_field: str,
        sort_direction: str,
    ) -> Tuple[int, List[Dict[str, Any]], str]:
        if sort_field not in fields:
            sort_field = default_sort_field

        sorted_records = sort_records(records, sort_by, sort_field, sort_direction)
        page_count = get_page_count(sorted_records)
        page_index = max(0, min(int(page_index or 0), page_count - 1))
        start_index = page_index * PAGE_SIZE
        end_index = min(start_index + PAGE_SIZE, len(sorted_records))
        page_records = sorted_records[start_index:end_index]
        status = (
            f"Page {page_index + 1} of {page_count} | "
            f"Showing {start_index + 1}-{end_index} of {len(sorted_records)}"
        )
        return page_index, page_records, status

    def row_values(
        page_index: int,
        sort_by: str,
        sort_field: str,
        sort_direction: str,
    ) -> List[Any]:
        page_index, page_records, status = get_page_records(
            page_index, sort_by, sort_field, sort_direction
        )
        updates = [page_index, gr.update(value=status)]
        page_start = page_index * PAGE_SIZE

        for slot in range(PAGE_SIZE):
            if slot >= len(page_records):
                updates.append(gr.update(value="", visible=False))
                updates.append(gr.update(value="", visible=False))
                for _ in fields:
                    updates.append(gr.update(value="", visible=False))
                continue

            record = page_records[slot]
            sorted_index = page_start + slot
            bleu_score = get_bleu_score(record)
            updates.append(
                gr.update(
                    value=format_record_header(
                        record, sorted_index, sort_by, sort_field
                    ),
                    visible=True,
                )
            )
            updates.append(
                gr.update(
                    value=f"**BLEU:** {bleu_score}",
                    visible=show_bleu and bleu_score is not None,
                )
            )
            for field in fields:
                updates.append(
                    gr.update(
                        value=display_value(record.get(field, "N/A")),
                        visible=True,
                    )
                )

        return updates

    def previous_page(
        page_index: int,
        sort_by: str,
        sort_field: str,
        sort_direction: str,
    ) -> List[Any]:
        return row_values((page_index or 0) - 1, sort_by, sort_field, sort_direction)

    def next_page(
        page_index: int,
        sort_by: str,
        sort_field: str,
        sort_direction: str,
    ) -> List[Any]:
        return row_values((page_index or 0) + 1, sort_by, sort_field, sort_direction)

    def first_page(
        sort_by: str,
        sort_field: str,
        sort_direction: str,
    ) -> List[Any]:
        return row_values(0, sort_by, sort_field, sort_direction)

    initial_page_index, initial_records, initial_status = get_page_records(
        0, SORT_ORIGINAL, default_sort_field, DIRECTION_DESC
    )

    with gr.Blocks(title="JSONL/JSON Record Viewer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# JSONL/JSON Record Viewer")
        gr.Markdown(f"**File:** {os.path.basename(jsonl_file)}")
        gr.Markdown(f"**Total Records:** {len(records)}")
        gr.Markdown(f"**Fields Displayed:** {', '.join(fields)}")

        page_state = gr.State(value=initial_page_index)

        with gr.Row():
            sort_by_dropdown = gr.Dropdown(
                choices=SORT_CHOICES,
                value=SORT_ORIGINAL,
                label="Sort by",
            )
            sort_field_dropdown = gr.Dropdown(
                choices=fields,
                value=default_sort_field,
                label="Length field",
            )
            sort_direction_dropdown = gr.Dropdown(
                choices=DIRECTION_CHOICES,
                value=DIRECTION_DESC,
                label="Direction",
            )

        with gr.Row():
            previous_button = gr.Button("Previous")
            page_status = gr.Markdown(value=initial_status)
            next_button = gr.Button("Next")

        row_headers = []
        row_bleu_scores = []
        row_field_textboxes = []

        for slot in range(PAGE_SIZE):
            record = initial_records[slot] if slot < len(initial_records) else None
            visible = record is not None
            sorted_index = slot

            with gr.Row():
                with gr.Column(scale=1):
                    row_headers.append(
                        gr.Markdown(
                            value=format_record_header(
                                record,
                                sorted_index,
                                SORT_ORIGINAL,
                                default_sort_field,
                            ) if record else "",
                            visible=visible,
                        )
                    )
                    bleu_score = get_bleu_score(record) if record else None
                    row_bleu_scores.append(
                        gr.Markdown(
                            value=f"**BLEU:** {bleu_score}" if bleu_score else "",
                            visible=show_bleu and bleu_score is not None,
                        )
                    )

                field_textboxes = []
                for field in fields:
                    with gr.Column(scale=2):
                        field_textboxes.append(
                            gr.Textbox(
                                label=field,
                                value=display_value(record.get(field, "N/A")) if record else "",
                                lines=4,
                                interactive=False,
                                visible=visible,
                            )
                        )
                row_field_textboxes.append(field_textboxes)

        page_outputs = [page_state, page_status]
        for slot in range(PAGE_SIZE):
            page_outputs.append(row_headers[slot])
            page_outputs.append(row_bleu_scores[slot])
            page_outputs.extend(row_field_textboxes[slot])

        previous_button.click(
            previous_page,
            inputs=[
                page_state,
                sort_by_dropdown,
                sort_field_dropdown,
                sort_direction_dropdown,
            ],
            outputs=page_outputs,
        )
        next_button.click(
            next_page,
            inputs=[
                page_state,
                sort_by_dropdown,
                sort_field_dropdown,
                sort_direction_dropdown,
            ],
            outputs=page_outputs,
        )
        for sort_control in (
            sort_by_dropdown,
            sort_field_dropdown,
            sort_direction_dropdown,
        ):
            sort_control.change(
                first_page,
                inputs=[
                    sort_by_dropdown,
                    sort_field_dropdown,
                    sort_direction_dropdown,
                ],
                outputs=page_outputs,
            )

    return interface


def main():
    parser = argparse.ArgumentParser(
        description="Create a Gradio interface for viewing JSONL/JSON records."
    )
    parser.add_argument(
        'jsonl_file',
        help="Path to the JSONL/JSON file to view"
    )
    parser.add_argument(
        'fields',
        nargs='+',
        help="Fields to display for each record"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help="Port to run the web server on (default: 7860)"
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help="Host to bind the server to (default: localhost)"
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help="Create a public link to share the interface"
    )

    args = parser.parse_args()

    if not os.path.exists(args.jsonl_file):
        print(f"Error: File '{args.jsonl_file}' does not exist.")
        return 1

    # Validate that fields exist in at least one record
    if is_jsonl_file(args.jsonl_file):
        test_records = read_jsonl_records(args.jsonl_file)
    else:
        test_records = read_json_records(args.jsonl_file)
    if not test_records:
        print(f"Error: No valid records found in '{args.jsonl_file}'.")
        return 1

    available_fields = set()
    for record in test_records:
        available_fields.update(record.keys())

    missing_fields = set(args.fields) - available_fields
    if missing_fields:
        print(f"Warning: Fields not found in any records: {', '.join(missing_fields)}")

    print(f"Starting JSONL/JSON viewer for '{args.jsonl_file}'")
    print(f"Fields to display: {', '.join(args.fields)}")
    print(f"Total records: {len(test_records)}")
    print(f"Open http://{args.host}:{args.port} in your browser")

    interface = create_main_interface(args.jsonl_file, args.fields)
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == '__main__':
    main()
