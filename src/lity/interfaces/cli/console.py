from __future__ import annotations

import sys

from lity.app.controller import AgentController


def run_console(ctrl: AgentController | None = None) -> None:
    controller = ctrl or AgentController()
    assistant_name = controller.assistant_name

    print("==============================================")
    print(f"   {assistant_name.upper()} - MODE CONSOLE")
    print("==============================================")

    try:
        while True:
            user_input = input("\nVous : ").strip()
            if not user_input:
                continue

            slash_result = controller.process_slash_command(user_input)
            if slash_result:
                if slash_result.get("action") == "quit":
                    print(f"\n{slash_result.get('message')}")
                    sys.exit(0)
                print(f"\n[SYSTÈME] {slash_result.get('message')}")
                continue

            result = controller.process_user_message_sync(user_input)
            _print_result(controller, result)
    except (EOFError, KeyboardInterrupt):
        print("\n\nAu revoir !")
        sys.exit(0)
    finally:
        controller.shutdown()


def _print_result(controller: AgentController, result: dict) -> None:
    result_type = result.get("type")
    if result_type == "intent_handled":
        print(f"\n[SYSTÈME] {result.get('message')}")
    elif result_type == "error":
        print(f"\n{controller.assistant_name} : {result.get('message')}")
    elif result_type in {"ai_response", "text"}:
        if result.get("system_notification"):
            print(f"\n[SYSTÈME] {result.get('system_notification')}")
        print(
            f"\n{controller.assistant_name} ({controller.engine.model}) : {result.get('content')}"
        )
        _review_file_blocks(controller, result)
    elif result_type and result_type.startswith("image_"):
        print(f"\n[IMAGE] {result.get('message', result.get('content'))}")
    else:
        print(f"\n[SYSTÈME] {result}")


def _review_file_blocks(controller: AgentController, result: dict) -> None:
    for block in result.get("create_blocks", []):
        print(f"\n[CRÉATION] {block['file_path']}")
        print(block["content"])
        if input("Créer ce fichier ? (o/n) : ").strip().lower() == "o":
            success, message = controller.apply_create_block(block)
            print(f"[SYSTÈME] {message}")
            if success:
                controller.files.load_file(block["file_path"])

    for block in result.get("edit_blocks", []):
        print(f"\n[MODIFICATION] {block['file_path']}")
        print("<<< SEARCH")
        print(block["search_content"])
        print("===")
        print(block["replace_content"])
        if input("Appliquer cette modification ? (o/n) : ").strip().lower() == "o":
            success, message = controller.apply_edit_block(block)
            print(f"[SYSTÈME] {message}")
            if success:
                controller.files.load_file(block["file_path"])
