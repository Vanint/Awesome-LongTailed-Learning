import json, os
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--file",
        help="json file to be converted",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--root",
        help="root path to save image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sp",
        help="save path for converted file ",
        type=str,
        required=False,
        default="."
    )

    args = parser.parse_args()
    return args

def convert(json_file, image_root):
    all_annos = json.load(open(json_file, 'r'))
    import ipdb; ipdb.set_trace()
    annos = all_annos['annotations']
    images = all_annos['images']
    new_annos = []

    print("Converting file {} ...".format(json_file))
    for anno, image in tqdm(zip(annos, images)):
        assert image["id"] == anno["id"]

        new_annos.append({"image_id": image["id"],
                          "im_height": image["height"],
                          "im_width": image["width"],
                          "category_id": anno["category_id"],
                          "fpath": os.path.join(image_root, image["file_name"])})
    num_classes = len(all_annos["categories"])
    return {"annotations": new_annos,
            "num_classes": num_classes}

if __name__ == "__main__":
    args = parse_args()
    converted_annos = convert(args.file, args.root)
    save_path = os.path.join(args.sp, "converted_" + os.path.split(args.file)[-1])
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)