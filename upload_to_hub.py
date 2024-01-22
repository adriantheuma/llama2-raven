from huggingface_hub import HfApi
api = HfApi()


api.upload_folder(
<<<<<<< HEAD
    folder_path="weights/",
    repo_id="unwilledset/raven-13b-chat-d8-no-tools",
=======
    folder_path="raven-13b-chat-d9-notools/",
    repo_id="adriantheuma/raven-lora-no-tools",
>>>>>>> a2d2503 (code cleanup prior to publication)
    repo_type="model",
    # ignore_patterns="checkpoint*", # Ignore all checkpoints
)