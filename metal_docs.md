Accelerated PyTorch training on Mac
Metal acceleration
PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration. This MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. The MPS framework optimizes compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family. The new mps device maps machine learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.

Requirements
Mac computers with Apple silicon or AMD GPUs
macOS 12.3 or later
Python 3.7 or later
Xcode command-line tools: xcode-select --install
Get started
You can use either Anaconda or pip. Please note that environment setup will differ between a Mac with Apple silicon and a Mac with Intel x86.

Use the PyTorch installation selector on the installation page to choose Preview (Nightly) for MPS device acceleration. The MPS backend support is part of the PyTorch 1.12 official release. The Preview (Nightly) build of PyTorch will provide the latest mps support on your device.

1. Set up
Anaconda
Apple silicon

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
x86

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
pip
You can use preinstalled pip3, which comes with macOS. Alternatively, you can install it from the Python website or the Homebrew package manager.

2. Install
Anaconda
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
Building from source
Building PyTorch with MPS support requires Xcode 13.3.1 or later. You can download the latest public Xcode release on the Mac App Store or the latest beta release on the Mac App Store or the latest beta release on the Apple Developer website. The USE_MPS environment variable controls building PyTorch and includes MPS support.

To build PyTorch, follow the instructions provided on the PyTorch website.

3. Verify
You can verify mps support using a simple Python script:

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
The output should show:

tensor([1.], device='mps:0')
Feedback
The MPS backend is in the beta phase, and we’re actively addressing issues and fixing bugs. To report an issue, use the GitHub issue tracker with the label “module: mps”.

Docs > MPS backend
MPS backend
mps device enables high-performance training on GPU for MacOS devices with Metal programming framework. It introduces a new device to map Machine Learning computational graphs and primitives on highly efficient Metal Performance Shaders Graph framework and tuned kernels provided by Metal Performance Shaders framework respectively.

The new MPS backend extends the PyTorch ecosystem and provides existing scripts capabilities to setup and run operations on GPU.

To get started, simply move your Tensor and Module to the mps device:

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
    