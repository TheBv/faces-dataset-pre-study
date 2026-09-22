import json
from typing import List, Literal
from tqdm import tqdm

from huggingface_hub import HfApi, HfFolder, Repository, upload_folder, create_repo
import os
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import DatasetDict


BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../.."))


def upload_to_hf_hub(model_dir: str, model_name: str, login_name: str, token: str):
    model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Push to Hugging Face with credentials
    model.push_to_hub(
        repo_id=f"{login_name}/{model_name}",
        use_auth_token=token
    )

    tokenizer.push_to_hub(
        repo_id=f"{login_name}/{model_name}",
        use_auth_token=token
    )

def upload_torch_nn_to_hub(model_dir: str, model_name: str, login_name: str, token: str):
    # Upload the entire folder to the Hub
    create_repo(
        repo_id=f"{login_name}/{model_name}",
        repo_type="model",
        token=token)
    upload_folder(
        folder_path=model_dir,
        repo_id=f"{login_name}/{model_name}",
        repo_type="model",
        token=token,
    )

def upload_hf_models_to_hf_hub_lang(lang: str, ignore: List[str], user_name: str, token: str):

    if lang == "" or lang == "en":
        cue_bert = f"{BP}/data/MODELS/CUE/gcn_cue_english_deberta_v2"
        scope_bert = f"{BP}/data/MODELS/SCOPE/gcn_scope_english_deberta_replace_v2"
        lang = "en"
    else:
        cue_bert = f"{BP}/data/MODELS/CUE{lang.upper()}/gcn_cue_{lang}_EuroBERT_EuroBERT-210m_v2"
        scope_bert = f"{BP}/data/MODELS/SCOPE{lang.upper()}/gcn_scope_{lang}_EuroBERT_EuroBERT-210m_v2"

    for ds in tqdm(["dt_neg", "socc", "bioscope_full", "bioscope_abstracts", "conan", "pb_foc", "sfu"]):
        if ds not in ignore:
            upload_to_hf_hub("_".join([cue_bert, ds]), f"cue-{lang}-{ds}", user_name, token)
            if ds != "pb_foc":
                upload_to_hf_hub("_".join([scope_bert, ds]), f"scope-{lang}-{ds}", user_name, token)


def upload_torch_models_to_hf_hub(lang: str,
                                  ignore: List[str],
                                  user_name: str,
                                  token: str,
                                  from_storage: bool = False,
                                  mode: Literal["cue", "scope", "both"] = "both"):

    if lang == "" or lang == "en":
        if from_storage:
            cue_bert = f"{BP}/data/MODELS/CUE_GAT/gcn_cue_english_debertabase_residual_gatv2_attentiongate"
            scope_bert = f"/storage/projects/hammerla/models/SCOPE_GAT/gcn_scope_english_debertabase_replace_residual_gatv2_attentiongate"
        else:
            cue_bert = f"{BP}/data/MODELS/CUE_GAT/gcn_cue_english_debertabase_residual_gatv2_attentiongate"
            scope_bert = f"{BP}/data/MODELS/SCOPE_GAT/gcn_scope_english_debertabase_replace_residual_gatv2_attentiongate"
        lang = "en"
    else:
        if from_storage:
            cue_bert = f"{BP}/data/MODELS/CUE_GAT{lang.upper()}/gcn_cue_{lang}_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate"
            scope_bert = f"/storage/projects/hammerla/models/SCOPE_GAT{lang.upper()}/gcn_scope_{lang}_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate"
        else:
            cue_bert = f"{BP}/data/MODELS/CUE_GAT{lang.upper()}/gcn_cue_{lang}_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate"
            scope_bert = f"{BP}/data/MODELS/SCOPE_GAT{lang.upper()}/gcn_scope_{lang}_EuroBERT_EuroBERT-210m_residual_gatv2_attentiongate"

    for ds in tqdm(["dt_neg", "socc", "bioscope_full", "bioscope_abstracts", "conan", "pb_foc", "sfu"]):
        if ds not in ignore:
            if mode == "cue" or mode == "both":
                upload_torch_nn_to_hub("_".join([cue_bert, ds]), f"cue-gat-{lang}-{ds}",user_name, token)
            if ds != "pb_foc":
                if mode == "scope" or mode == "both":
                    upload_torch_nn_to_hub("_".join([scope_bert, ds]), f"scope-gat-{lang}-{ds}", user_name, token)


def upload_ds_hfhub(ds_dict_path: str, login_name: str, token: str, atype: Literal["cue", "scope"]):

    # Push to Hugging Face with credentials
    ds_dict = DatasetDict.load_from_disk(ds_dict_path)
    # print(ds_dict)
    for key in ds_dict.keys():
        try:
            ds_dict[key] = ds_dict[key].remove_columns(["sentence_strings", "id"])
        except:
            pass
        if atype == "cue":
            try:
                ds_dict[key] = ds_dict[key].remove_columns(["event_masks", "focus_masks", "scope_masks"])
            except:
                pass
        elif atype == "scope":
            try:
                ds_dict[key] = ds_dict[key].remove_columns(["event_masks", "focus_masks"])
            except:
                pass
        else:
            raise NotImplementedError

    print(ds_dict)
    ds_dict.push_to_hub(f"{login_name}/{ds_dict_path.split('/')[-1]}",
                        private=False,
                        token = token
    )



if __name__ == "__main__":
    """
    From storage: de, es, zh, hi, jap, ru
    """
    with open(f"{BP}/data/HF-CREDENTIALS/hf_login.json", "r") as f:
        hf_login = json.load(f)
    for lang in ["de", "hi", "ar", "it", "es", "ru", "zh", "jap", "fr", "nl", "en"]:
    # for lang in ["en"]:
        if lang in ["de", "es", "zh", "hi", "jap", "ru"]:
            from_storage = True
        else:
            from_storage = False
        """upload_hf_models_to_hf_hub_lang(lang=lang,
                                        ignore=["pb_foc", "bioscope_abstracts"],
                                        user_name=hf_login["user_name_dneg"],
                                        token=hf_login["token_dneg"])
        upload_torch_models_to_hf_hub(lang=lang,
                                      ignore=["pb_foc", "bioscope_abstracts"],
                                      from_storage=from_storage,
                                      mode="both",
                                      user_name=hf_login["user_name_dneg"],     
                                      token=hf_login["token_dneg"])"""
        try:
            upload_ds_hfhub(f"{BP}/data/HF-DATASETS/NEG-split-cleaned-cue-{lang}" if lang != "en" else f"{BP}/data/HF-DATASETS/NEG-split-cleaned-cue", hf_login["user_name_dneg"], hf_login["token_dneg"], "cue")
            upload_ds_hfhub(f"{BP}/data/HF-DATASETS/NEG-split-cleaned-scope-{lang}" if lang != "en" else f"{BP}/data/HF-DATASETS/NEG-split-cleaned-scope", hf_login["user_name_dneg"], hf_login["token_dneg"], "scope")
        except Exception as e:
            print(lang)
            print(e)

