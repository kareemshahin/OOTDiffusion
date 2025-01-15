import modal

app = modal.App("baxter-try-on")
#dockerfile_image = modal.Image.from_dockerfile("Dockerfile")
dockerfile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"])
    .pip_install("torch==2.0.1")
    .pip_install("torchvision==0.15.2")
    .pip_install("torchaudio==2.0.2")
    .pip_install_from_requirements("./requirements.txt")
    .pip_install("huggingface_hub==0.25.2")
    .add_local_dir(".", remote_path="/app")
)
checkpoints = modal.Volume.from_name("ootd-checkpoints")
viton_output = modal.Volume.from_name("viton_output")
input_images = modal.Volume.from_name("input_images")

volume_map = {
    "/app/checkpoints": checkpoints,
    "/app/run/outputs": viton_output,
    "/app/run/input_images": input_images,
}
WORKING_DIR = "/app/run"

@app.function(gpu="A100", image=dockerfile_image, volumes=volume_map)
def try_on(
    model_path="/app/run/examples/model/01008_00.jpg",
    cloth_path="/app/run/examples/garment/00055_00.jpg",
    category="full", scale="2.0", sample="1"
):
    import subprocess

    model_type = 'dc'
    input_category = '2'

    if category in ['lower', 'upper']:
        model_type = 'hd'
        input_category = '0' if category == 'upper' else '1'

    cmdz = [
        "python",
        "run_ootd.py",
        "--model_path",
        model_path,
        "--cloth_path",
        cloth_path,
        "--category",
        input_category,
        "--model_type",
        model_type,
        "--scale",
        scale,
        "--sample",
        sample,
    ]
    print(*cmdz)

    retcode = subprocess.run(cmdz, cwd=WORKING_DIR)
    print(retcode)
    #print(f"model_path={model_path},cloth_path={cloth_path},category={category}({model_type},{input_category})")


#@app.local_entrypoint()
#def main(
#    model_path="/app/run/examples/model/01008_00.jpg",
#    cloth_path="/app/run/examples/garment/00055_00.jpg",
#    category="full", scale="2.0", sample="1"
#):
#    try_on.remote(model_path, cloth_path, category, scale, sample)
