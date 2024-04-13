import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "include_files": ["selfie_multiclass_256x256.xml", "selfie_multiclass_256x256.bin", "sample.png"]
}

# base="Win32GUI" should be used only for Windows GUI app
base = "console"

setup(
    name="run_openvino_demo",
    version="0.1",
    author="Intel Corporation (ryan.loney@intel.com)",
    description="OpenVINO Segmentation Demo",
    options={"build_exe": build_exe_options},
    executables=[Executable("run_openvino_demo.py", base=base, icon="ov_logo.ico")],
)
