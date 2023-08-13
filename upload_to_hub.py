from huggingface_hub import HfApi
api = HfApi()


api.upload_folder(
    folder_path="weights/",
    repo_id="unwilledset/raven-13b-chat-d6",
    repo_type="model",
    ignore_patterns="checkpoint*", # Ignore all checkpoints
)