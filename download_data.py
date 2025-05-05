# download_data.py  ── run ONCE to build data/train/  and data/val/
import pathlib, json, tqdm, base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset

root = pathlib.Path("data")
splits = {"train": "train", "validation": "val"}      # to rename "validation" → val/

for split_hf, split_local in splits.items():
    outdir = root / split_local
    (outdir / "images").mkdir(parents=True, exist_ok=True)

    # load  (add trust_remote_code=True to skip the y/N prompt next time)
    ds = load_dataset(
        "MMInstruction/M3IT",
        "image-paragraph-captioning",
        split=split_hf,
        trust_remote_code=True,    # avoid interactive prompt
        streaming=True             # iterate without loading everything in RAM
    )

    jsonl = open(outdir / "stories.jsonl", "w")
    for i, rec in enumerate(tqdm.tqdm(ds, desc=f"{split_hf} split")):
        # decode first image in list
        b64_str = rec["image_base64_str"][0]
        img = Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")
        fn = f"{i:06d}.jpg"
        img.save(outdir / "images" / fn, "JPEG")

        # grab paragraph text
        paragraph = rec["outputs"].strip()
        prompt = paragraph.split(".")[0].strip() + "."

        # write JSON‑Lines record
        json.dump(
            {
                "prompt": prompt,
                "story":  paragraph,
                "image_path": f"images/{fn}"
            },
            jsonl
        )
        jsonl.write("\n")

    jsonl.close()
