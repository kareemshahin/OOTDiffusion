import modal

app = modal.App("baxter-try-on")
#dockerfile_image = modal.Image.from_dockerfile("Dockerfile")
dockerfile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "libavif-dev", "libaom-dev"])
    .pip_install("torch==2.0.1")
    .pip_install("torchvision==0.15.2")
    .pip_install("torchaudio==2.0.2")
    .pip_install_from_requirements("./requirements.txt")
    .pip_install("huggingface_hub==0.25.2")
    .pip_install("requests")
    .workdir("/app/run")
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

@app.function(gpu="a10g", image=dockerfile_image, volumes=volume_map)
def try_on(
    model_path="/app/run/examples/model/01008_00.jpg",
    cloth_path="/app/run/examples/garment/00055_00.jpg",
    category="full", scale=2.0, sample=1
):
    from ootd_generator import OOTDGenerator

    model_type = 'dc'
    input_category = 2

    if category in ['lower', 'upper']:
        model_type = 'hd' if category == 'upper' else 'dc'
        input_category = 0 if category == 'upper' else 1

    scale = float(scale)
    input_category = int(input_category)
    sample = int(sample)

    ootd_generator = OOTDGenerator(
        gpu_id=0,
        model_path=model_path,
        cloth_path=cloth_path,
        model_type=model_type,
        category=input_category,
        scale=scale,
        step=20,
        sample=sample,
        seed=-1
    )

    img_data = ootd_generator.generate_images()

    print(img_data)
    return { "images": img_data }
