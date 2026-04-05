from typing import Dict, List


REPAIR_RULES = {
    "missing": "Replace missing component and verify BOM compliance. Check for tombstoning or skew during reflow. Perform ICT test post-repair.",
    "short": "Remove solder bridge using hot air station or desoldering wick. Clean pads thoroughly. Verify isolation with multimeter before reassembly.",
    "open circuit": "Reconnect open trace using conductive epoxy or jumper wire. For critical nets, use proper trace repair techniques. Validate signal integrity.",
    "shifted": "Realign component and reflow solder joints. Check for pad damage or lifted pads. Verify component orientation and pin alignment.",
    "misaligned": "Correct component alignment and perform touch-up soldering. Check solder joint quality under magnification. Verify electrical connectivity.",
    "solder_bridge": "Remove excess solder with wick or hot air. Clean flux residue. Inspect adjacent pins for collateral damage.",
    "scratch": "Evaluate scratch depth - if conductive layers exposed, repair may be required. Document for process improvement.",
}


def suggest_repairs(defects: List[Dict]) -> List[Dict]:
    suggestions = []
    if not defects:
        return [{"defect": "none", "suggestion": "No repair needed. PCB passed inspection. Proceed with functional testing and programming."}]

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
            suggestion = "Perform targeted visual inspection and rework based on IPC-A-610 criteria. Document findings for process improvement."

        key_tuple = (label, suggestion)
        if key_tuple not in seen:
            suggestions.append({"defect": label, "suggestion": suggestion})
            seen.add(key_tuple)

    return suggestions
