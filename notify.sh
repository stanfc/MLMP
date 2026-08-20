#!/bin/bash
# Push a notification to your phone via ntfy.sh.
# Subscribe once in the ntfy app to topic:  mlmp-tta-9361a736
# Usage: bash notify.sh "your message"  [title]
TOPIC="mlmp-tta-9361a736"
MSG="${1:-run finished}"
TITLE="${2:-MLMP run}"
curl -s -H "Title: $TITLE" -d "$MSG" "https://ntfy.sh/$TOPIC" >/dev/null && echo "[notify] sent to $TOPIC"
