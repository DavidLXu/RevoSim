"""Use only Isaac Lab core and the rendering/physics extensions RevoSim needs."""

import json, tempfile
from pathlib import Path

_experiences = []


def configure(args):
    if getattr(args, "experience", ""):
        return args
    import isaaclab

    source = Path(isaaclab.__file__).resolve().parents[2]
    template = (Path(__file__).parent / "apps/revosim.kit").read_text()
    template = template.replace('"${app}/../source"', json.dumps(str(source)))
    tmp = tempfile.TemporaryDirectory(prefix="revosim-kit-")
    _experiences.append(tmp)
    path = Path(tmp.name) / "revosim.kit"
    path.write_text(template)
    args.experience = str(path)
    return args
