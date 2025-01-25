# Video Frame Interpolation

This project implements video frame interpolation using the U-Net architecture and pre-trained Super SloMo weights. Frame interpolation involves generating intermediate frames between existing video frames to achieve a higher frame rate, resulting in smoother playback.

## Features

- Utilizes U-Net architecture for video frame interpolation.
- Employs pre-trained Super SloMo weights for accurate and efficient interpolation.
- Generates interpolated frames for any input video.
- Outputs the interpolated video at the desired frames per second (FPS).

## Requirements

Ensure you have the following installed on your system:

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To interpolate video frames and generate a smoother video, use the following command:

```bash
python main.py --ofps 120 --ivideo "Test.mp4" --ovideo "Result.mp4" --fdir "output"
```

### Arguments

1. `--ofps` : Desired output frames per second (FPS). Specify the target FPS for the interpolated video.
2. `--ivideo` : Path to the input video file. Provide the name or path of the video to be interpolated.
3. `--ovideo` : Name of the final interpolated video file. Specify the name of the output video file (e.g., `Result.mp4`).
4. `--fdir` : Directory to store output frames from the original video. Provide the name of the directory where intermediate frames will be saved.

### Example

```bash
python main.py --ofps 60 --ivideo "example.mp4" --ovideo "output.mp4" --fdir "frames"
```
This command will:
- Interpolate the frames of `example.mp4` to achieve 60 FPS.
- Save the resulting video as `output.mp4`.
- Store the extracted frames in the `frames` directory.

## Project Structure

- `main.py` : The main script for running the interpolation.
- `models/` : Contains the U-Net architecture and Super SloMo weights.
- `requirements.txt` : Lists all the dependencies required for the project.
- `README.md` : Documentation for the project.

## How It Works

1. **Frame Extraction**: Extracts frames from the input video.
2. **Interpolation**: Applies the U-Net architecture with Super SloMo weights to generate intermediate frames.
3. **Video Reconstruction**: Combines the interpolated frames into a new video with the desired FPS.

## Acknowledgments

- This project utilizes pre-trained weights from the [Super SloMo](https://github.com/avinashpaliwal/Super-SloMo) repository.
- The U-Net architecture was adapted for efficient frame interpolation.

## Contributing

Feel free to contribute by submitting issues or pull requests to improve this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

For any questions or suggestions, please open an issue in this repository.
