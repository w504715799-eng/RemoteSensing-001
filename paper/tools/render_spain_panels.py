"""Plot four protocol-selected ROIs from a verified deployment; CPU/cache only.

Usage: python render_spain_panels.py DEPLOYMENT_DIRECTORY
The directory contains code/, packages/ and the completed run/.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    b = Path(sys.argv[1]).resolve()
    c = b / "code"
    sys.path.insert(0, str(c / "src"))
    import trustsr

    require(
        Path(trustsr.__file__).resolve() == c / "src/trustsr/__init__.py",
        "panel evidence validation failed",
    )
    from trustsr.artifacts.predictions import PredictionCache, build_identity, tensor_sha256
    from trustsr.data.spain_package import decode_package
    from trustsr.evaluation.external_protocol import load_frozen
    from trustsr.evaluation.external_replay import K5_SLOTS
    from trustsr.risk.local import ensemble_variance_score, local_l1_risk
    from trustsr.risk.neighborhood import neighborhood_variance_score
    from trustsr.risk.proxies import lr_reprojection_l1_score

    sha = "798200c5c8b2cd9102755202810e8161832064de686a81b41c02776ddaf85525"
    p = load_frozen((c / "paper/protocols/spain-external-frozen-v1.json").read_bytes(), sha, c)
    m = json.loads((b / "run/manifest.json").read_bytes())
    raw = (b / "run/science.json").read_bytes()
    expected_science = "f09fc42555cb00deac5125e478b61817200de9a796bd668e5abcc15fbfb07fe4"
    require(
        hashlib.sha256(raw).hexdigest() == m["science_sha256"] == expected_science,
        "panel evidence validation failed",
    )
    require(m["protocol_sha256"] == sha, "panel evidence validation failed")
    s = json.loads(raw)
    require(s["protocol_sha256"] == sha, "panel evidence validation failed")
    cache = PredictionCache(b / "run/cache")
    torch.set_num_threads(1)
    fig, axes = plt.subplots(4, 7, figsize=(14, 8), layout="constrained")
    titles = [
        "LR RGB",
        "LDSR RGB",
        "Reference RGB",
        "LR score",
        "K5 variance",
        "Neighborhood w3",
        "R9 risk",
    ]
    selected = []
    r = 0
    for subset, spec in sorted(p["science"]["subsets"].items()):
        pairs, receipt = decode_package(
            b / "packages" / spec["package_filename"],
            expected_sha256=spec["package_sha256"],
            expected_size=spec["package_size"],
            members=spec["members"],
        )
        require(receipt == s["input_receipts"][subset], "panel evidence validation failed")
        byid = {pair.sample_id: pair for pair in pairs}
        rows = {row["roi"]: row for row in s["subsets"][subset]["rows"]}
        for roi in sorted(byid)[:2]:
            pair = byid[roi]
            row = rows[roi]
            selected.append({"subset": subset, "roi": roi})
            require(
                tensor_sha256(pair.lr) == row["inputs"]["lr_sha256"]
                and tensor_sha256(pair.hr) == row["inputs"]["hr_sha256"],
                "panel input identity mismatch",
            )
            available = {}
            for slot in K5_SLOTS:
                x = cache.get(
                    build_identity(p["science"]["provenances"][slot], pair.source, roi, pair.lr)
                )
                require(
                    (x is not None) == (row["predictions"][slot]["status"] == "valid"),
                    "published prediction availability changed",
                )
                if x is not None:
                    require(
                        tensor_sha256(x) == row["predictions"][slot]["sha256"],
                        "panel evidence validation failed",
                    )
                    available[slot] = x
            center = available.get("ldsr_3407")
            maps = [pair.lr, center, pair.hr, None, None, None, None]
            if center is not None:
                maps[3] = lr_reprojection_l1_score(center, pair.lr)
                maps[6] = local_l1_risk(center, pair.hr, window=9)
                if all(slot in available for slot in K5_SLOTS):
                    samples = torch.stack([available[slot] for slot in K5_SLOTS])
                    maps[4] = ensemble_variance_score(samples)
                    maps[5] = neighborhood_variance_score(samples, window=3)
            for col, value in enumerate(maps):
                ax = axes[r, col]
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(titles[col], fontsize=10)
                if col == 0:
                    ax.set_ylabel(subset.replace("spain_", "") + " " + roi, fontsize=9)
                if value is None:
                    ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                    continue
                a = value.detach().cpu().numpy()
                if col < 3:
                    ax.imshow(np.moveaxis(np.clip(a[:3], 0, 1) ** (1 / 2.2), 0, -1))
                else:
                    vmax = {
                        3: 0.05,
                        4: 4 * p["science"]["transfer_threshold"],
                        5: 4 * p["science"]["transfer_threshold"],
                        6: 0.1,
                    }[col]
                    im = ax.imshow(a, vmin=0, vmax=vmax, cmap="magma")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            r += 1
    output = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else b / "figures"
    output.mkdir(exist_ok=False)
    fig.savefig(output / "spain_fixed_examples.png", dpi=160)
    fig.savefig(output / "spain_fixed_examples.pdf", dpi=160)
    plt.close(fig)
    record = {
        "protocol_sha256": sha,
        "science_sha256": m["science_sha256"],
        "selected": selected,
        "rgb_gamma": 2.2,
        "scalar_display_maxima": {
            "lr": 0.05,
            "k5": 4 * p["science"]["transfer_threshold"],
            "neighborhood": 4 * p["science"]["transfer_threshold"],
            "R9": 0.1,
        },
        "display_clipping_only": True,
        "new_inference": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in output.iterdir()},
    }
    with (output / "panel-receipt.json").open("x") as f:
        json.dump(record, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(record))


if __name__ == "__main__":
    main()
