import os, glob, json

source = "/home/dolan/cse676/project/data/chest_xray/train/"  # root with NORMAL/ and PNEUMONIA/
classes = ["NORMAL", "PNEUMONIA"]  # 0, 1

labels = []
for class_idx, cls in enumerate(classes):
    pattern = os.path.join(source, cls, "*")
    for fname in sorted(glob.glob(pattern)):
        if not os.path.isfile(fname):
            continue
        rel = os.path.relpath(fname, source).replace("\\", "/")
        labels.append([rel, class_idx])

meta = {"labels": labels}
out_path = os.path.join(source, "dataset.json")
with open(out_path, "w") as f:
    json.dump(meta, f)
print(f"Wrote {len(labels)} labels to {out_path}")