import modal

app = modal.App("baxter-try-on")
dockerfile_image = modal.Image.from_dockerfile("Dockerfile")
vol = modal.Volume.from_name("ootd-checkpoints")

@app.function(image=dockerfile_image, volumes={"/app/checkpoints": vol})
def test():
    import subprocess

    cmdz = [
        "python",
        "/app/run/run_ootd.py",
        "--model_path",
        "/app/run/examples/model/01008_00.jpg",
        "--cloth_path",
        "/app/run/examples/garment/00055_00.jpg",
        "--scale",
        "2.0",
        "--sample",
        "4"
    ]

    retcode = subprocess.run(cmdz)
    print(retcode)

