"""Conversation logger to persist full conversation history to disk."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ConversationLogger:
    """Logs conversation history to JSON files in data/conversations/."""
    
    def __init__(self, conversations_dir: str = "./data/conversations"):
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_path: Path | None = None
        self.current_conversation_id: str | None = None
        self.turn_count: int = 0
    
    def start_conversation(self, conversation_id: str) -> None:
        """Start a new conversation log."""
        self.current_conversation_id = conversation_id
        self.turn_count = 0
        
        # Create log file: YYYY-MM-DD_HHMMSS_<conversation_id>.json
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{timestamp}_{conversation_id[:8]}.json"
        self.current_log_path = self.conversations_dir / filename
        
        # Initialize log file
        initial_data = {
            "conversation_id": conversation_id,
            "started_at": datetime.now().isoformat(),
            "turns": []
        }
        
        with open(self.current_log_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    def log_turn(
        self,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Log a conversation turn."""
        if not self.current_log_path:
            return
        
        self.turn_count += 1
        
        turn_data = {
            "turn_index": self.turn_count - 1,  # 0-indexed to match observer
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_response,
        }
        
        if metadata:
            turn_data["metadata"] = metadata
        
        # Read existing log
        with open(self.current_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        
        # Append turn
        log_data["turns"].append(turn_data)
        log_data["last_updated"] = datetime.now().isoformat()
        log_data["total_turns"] = len(log_data["turns"])
        
        # Write back
        with open(self.current_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def end_conversation(self) -> None:
        """Mark conversation as ended."""
        if not self.current_log_path:
            return
        
        with open(self.current_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        
        log_data["ended_at"] = datetime.now().isoformat()
        log_data["final_turn_count"] = self.turn_count
        
        with open(self.current_log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        self.current_log_path = None
        self.current_conversation_id = None
        self.turn_count = 0
