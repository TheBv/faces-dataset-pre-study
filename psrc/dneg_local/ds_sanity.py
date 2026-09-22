from functools import partial

from .data import load_hf_dataset, remove_cue_from_scope, replace_cue_special_token


class DsSanity:
    @staticmethod
    def check_scope_ds(variant: str = "reference", split: str = "train", n: int = 10):
        """Print scope examples exactly as the plain scope model sees them."""
        cue_special_token = "[CUE]"
        ds = load_hf_dataset("scope", variant)[split]
        ds = ds.map(remove_cue_from_scope)
        ds = ds.map(partial(replace_cue_special_token, cue_special_token=cue_special_token))

        print("=== EXAMPLES ===")
        for example in ds.select(range(min(n, len(ds)))):
            print(example["sentence_id"])
            print(" ".join(f"{t}/S" if s else t for t, s in zip(example["tokens"], example["scope_masks"])))
            print("================")


if __name__ == "__main__":
    DsSanity.check_scope_ds()
