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
    #.pip_install("numpy==1.24.4")
    #.pip_install("scipy==1.10.1")
    #.pip_install("scikit-image==0.21.0")
    #.pip_install("opencv-python==4.7.0.72")
    #.pip_install("pillow==9.4.0")
    #.pip_install("diffusers==0.24.0")
    #.pip_install("transformers==4.36.2")
    #.pip_install("accelerate==0.26.1")
    #.pip_install("matplotlib==3.7.4")
    #.pip_install("tqdm==4.64.1")
    #.pip_install("gradio==4.16.0")
    #.pip_install("config==0.5.1")
    #.pip_install("einops==0.7.0")
    #.pip_install("onnxruntime==1.16.2")
)
checkpoints = modal.Volume.from_name("ootd-checkpoints")
viton_output = modal.Volume.from_name("viton_output")
input_images = modal.Volume.from_name("input_images")

volume_map = {
    "/app/checkpoints": checkpoints,
    "/app/run/outputs": viton_output,
    "/app/run/inputs": input_images,
}
WORKING_DIR = "/app/run"

@app.function(gpu="A100", image=dockerfile_image, volumes=volume_map)
def test():
    import subprocess

    cmdz = [
        "python",
        "run_ootd.py",
        "--model_path",
        "/app/run/examples/model/01008_00.jpg",
        "--cloth_path",
        "/app/run/examples/garment/00055_00.jpg",
        "--scale",
        "2.0",
        "--sample",
        "4"
    ]

    retcode = subprocess.run(cmdz, cwd=WORKING_DIR)
    print(retcode)


@app.local_entrypoint()
def main():
    test.remote()
