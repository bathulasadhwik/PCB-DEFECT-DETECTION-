from typing import Dict, List


REPAIR_RULES = {
    "missing": "Replace the missing component and verify orientation and BOM value before reflow.",
    "short": "Remove solder bridge using hot air or wick, then inspect pad spacing under magnification.",
    "open circuit": "Reconnect the open trace/joint and validate continuity using a multimeter.",
    "shifted": "Realign the component and rework solder joints to center pads.",
}


def suggest_repairs(defects: List[Dict]) -> List[Dict]:
    suggestions = []
    if not defects:
        return [{"defect": "none", "suggestion": "No repair needed. PCB passed inspection."}]

    seen = set()
    for defect in defects:
        label = defect.get("label", "unknown")
        label_l = label.lower()

        suggestion = None
        for key, text in REPAIR_RULES.items():
            if key in label_l:
                suggestion = text
                break

        if suggestion is None:
            suggestion = "Perform targeted visual inspection and rework based on IPC-A-610 criteria."

        key_tuple = (label, suggestion)
        if key_tuple not in seen:
            suggestions.append({"defect": label, "suggestion": suggestion})
            seen.add(key_tuple)

    return suggestions
