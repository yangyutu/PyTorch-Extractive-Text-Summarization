import wandb

run = wandb.init()
artifact = run.use_artifact(
    "yangyutu/Text-Summarization/model-3587qhuc:v5", type="model"
)
artifact_dir = artifact.download()
