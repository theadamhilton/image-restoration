# Image Restoration using Stable Diffusion

This project demonstrates how to use a pretrained Stable Diffusion model for image restoration tasks. The model is loaded using the Hugging Face `diffusers` library and is used to restore a damaged image by performing inpainting.

## Requirements

- Python 3.7+
- CUDA-enabled GPU (optional but recommended for faster inference)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/image-restoration-stable-diffusion.git
    cd image-restoration-stable-diffusion
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load a pretrained Stable Diffusion model and restore a damaged image:

    ```python
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    import requests
    import numpy as np
    from io import BytesIO

    # Load the pretrained model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")  # Use GPU for faster inference

    # Load a damaged image
    url = "https://example.com/damaged_image.jpg"
    response = requests.get(url)
    damaged_image = Image.open(BytesIO(response.content))

    # Perform inpainting to restore the image
    prompt = "A restored version of the damaged image"
    restored_image = pipe(prompt, image=damaged_image).images[0]

    # Save the restored image
    restored_image.save("restored_image.jpg")
    ```

2. Customize the prompt and input image as needed to achieve the desired restoration results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
